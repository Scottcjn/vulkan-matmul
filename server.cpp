/*
 * vulkan_matmul_server.cpp  v3 — Single-Submit Fast Path
 *
 * Weight matrices (B) are cached in Navi 12's 8GB VRAM.
 * Per-token cost: only send activation A (~28KB) + get result C (~28KB).
 * Weight upload happens ONCE per unique weight tensor, then LRU-cached.
 *
 * v3 adds compute_cached_fast(): for warm cache hits (a_only=1), the entire
 * sequence (upload A → dispatch → download C) runs in ONE command buffer with
 * ONE vkQueueSubmit+vkWaitForFences.  v2 used two submits (~43ms floor);
 * v3 targets ~22ms floor by eliminating the extra fence-wait round-trip.
 *
 * Fast path condition: szA + szC ≤ stageSize (always true for M=1..~2000)
 *
 * Protocol v2 (32-byte header):
 *   magic(4) M(4) N(4) K(4) type(4) a_only(4) key_lo(4) key_hi(4)
 *
 *   type=0  Plain matmul    → send A + B, get C        (no caching)
 *   type=1  Cached matmul:
 *     a_only=0 → send A + B (client says "weight may not be cached")
 *     a_only=1 → send A only (client says "weight should be in cache")
 *
 *   Response (16 bytes): magic(4) status(4) M(4) N(4)  then C[M*N*4]
 *   status=0  OK
 *   status=2  NEED_WEIGHT — client must retry with a_only=0
 *
 * Build:  cmake -B build . && cmake --build build -j16
 * Run:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.ppc64le.json \
 *   ./build/vulkan_matmul_server 8097
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#define MAGIC_VK       0x564B4D54u
#define MAX_DIM        65536
#define STAGING_MB     240          /* host-visible staging (≤ 256MB BAR) */
#define CACHE_VRAM_MB  7000         /* 7 GB weight cache in device-local VRAM */
#define MAX_CACHE      512          /* max cached weight tensors */
#define STATUS_OK      0u
#define STATUS_NEED_W  2u           /* tell client: send weight */
#define TYPE_PLAIN     0u
#define TYPE_CACHED    1u

/* ────────────────── helpers ────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
static void die(const char *m) { fprintf(stderr,"FATAL: %s\n",m); exit(1); }

static int recv_all(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t*)buf; size_t left = n;
    while (left) { ssize_t r = recv(fd,p,left,MSG_WAITALL); if(r<=0) return -1; p+=r; left-=r; }
    return 0;
}
static int send_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t*)buf; size_t left = n;
    while (left) { ssize_t r = send(fd,p,left,0); if(r<=0) return -1; p+=r; left-=r; }
    return 0;
}
static uint8_t *load_file(const char *path, size_t *sz) {
    FILE *f = fopen(path,"rb"); if(!f){perror(path);return NULL;}
    fseek(f,0,SEEK_END); *sz=ftell(f); rewind(f);
    uint8_t *b = (uint8_t*)malloc(*sz); fread(b,1,*sz,f); fclose(f);
    return b;
}

/* ────────────────── Vulkan state ────────────────── */

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queueFamily;

    VkShaderModule        shaderModule;
    VkDescriptorSetLayout dsetLayout;
    VkPipelineLayout      pipeLayout;
    VkPipeline            pipeline;

    VkCommandPool    cmdPool;
    VkCommandBuffer  cmdBuf;
    VkDescriptorPool dPool;
    VkDescriptorSet  dSet;
    VkFence          fence;

    /* Staging buffer (host-visible, BAR-mapped, persistent) */
    VkBuffer       stageBuf;
    VkDeviceMemory stageMem;
    void          *stageMapped;
    size_t         stageSize;

    /* Dynamic device-local buffers for A and C (re-alloc as needed) */
    VkBuffer       bufA, bufC;
    VkDeviceMemory memA, memC;
    size_t         capA, capC;
} VkCtx;

/* ────────────────── Weight cache ────────────────── */

typedef struct {
    uint64_t       key;           /* (uint64_t)src0->data in llama.cpp */
    VkBuffer       buf;
    VkDeviceMemory mem;
    size_t         bytes;
    uint32_t       K, N;          /* weight shape: [K, N] */
    uint64_t       last_used;     /* LRU sequence counter */
    int            valid;
} WtEntry;

static WtEntry  g_wt[MAX_CACHE];
static size_t   g_wt_used_bytes = 0;
static uint64_t g_wt_seq        = 0;
static size_t   g_wt_budget     = (size_t)CACHE_VRAM_MB * 1024 * 1024;

static VkCtx g;

/* Cache stats */
static uint64_t g_cache_hits   = 0;
static uint64_t g_cache_misses = 0;
static uint64_t g_evictions    = 0;

/* ────────────────── Memory helpers ────────────────── */

static uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    die("No suitable memory type"); return 0;
}

static void allocDevBuf(VkDeviceSize sz, VkBufferUsageFlags usage,
                        VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sz;
    bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(g.device, &bi, NULL, buf);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g.device, *buf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = findMemType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g.device, &ai, NULL, mem) != VK_SUCCESS) {
        fprintf(stderr, "[vk] vkAllocateMemory failed for %zu MB\n", (size_t)sz>>20);
        *mem = VK_NULL_HANDLE;
        return;
    }
    vkBindBufferMemory(g.device, *buf, *mem, 0);
}

static void ensureDevBuf(size_t needed, VkBufferUsageFlags usage,
                         VkBuffer *buf, VkDeviceMemory *mem, size_t *cap) {
    if (needed <= *cap) return;
    if (*cap > 0) {
        vkDeviceWaitIdle(g.device);
        vkDestroyBuffer(g.device, *buf, NULL);
        vkFreeMemory(g.device, *mem, NULL);
    }
    size_t nc = ((needed + (1<<20)-1) >> 20) << 20;
    allocDevBuf((VkDeviceSize)nc, usage, buf, mem);
    *cap = (*mem != VK_NULL_HANDLE) ? nc : 0;
}

/* One-shot: stage → device copy via command buffer (blocking) */
static void stageToDevice(VkBuffer dst, VkDeviceSize dstOff,
                           VkDeviceSize srcOff, VkDeviceSize size) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g.cmdBuf, &bi);
    VkBufferCopy cp = {srcOff, dstOff, size};
    vkCmdCopyBuffer(g.cmdBuf, g.stageBuf, dst, 1, &cp);
    VkBufferMemoryBarrier bar = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer = dst; bar.offset = dstOff; bar.size = size;
    vkCmdPipelineBarrier(g.cmdBuf,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,0,NULL,1,&bar,0,NULL);
    vkEndCommandBuffer(g.cmdBuf);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g.cmdBuf;
    vkQueueSubmit(g.queue, 1, &si, g.fence);
    vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g.device, 1, &g.fence);
}

/* Upload large buffer to device-local in chunks ≤ stageSize */
static int uploadChunked(VkBuffer dst, const float *src, size_t totalBytes) {
    size_t off = 0;
    while (off < totalBytes) {
        size_t chunk = totalBytes - off;
        if (chunk > g.stageSize) chunk = g.stageSize;
        memcpy(g.stageMapped, (const uint8_t*)src + off, chunk);
        stageToDevice(dst, (VkDeviceSize)off, 0, (VkDeviceSize)chunk);
        off += chunk;
    }
    return 0;
}

/* ────────────────── Weight cache management ────────────────── */

static WtEntry *cache_lookup(uint64_t key) {
    for (int i = 0; i < MAX_CACHE; i++)
        if (g_wt[i].valid && g_wt[i].key == key) return &g_wt[i];
    return NULL;
}

static void cache_evict_lru(size_t needBytes) {
    while (g_wt_used_bytes + needBytes > g_wt_budget) {
        /* Find LRU entry */
        int lru = -1;
        uint64_t lru_seq = UINT64_MAX;
        for (int i = 0; i < MAX_CACHE; i++) {
            if (g_wt[i].valid && g_wt[i].last_used < lru_seq) {
                lru_seq = g_wt[i].last_used;
                lru = i;
            }
        }
        if (lru < 0) break;
        vkDeviceWaitIdle(g.device);
        vkDestroyBuffer(g.device, g_wt[lru].buf, NULL);
        vkFreeMemory(g.device, g_wt[lru].mem, NULL);
        g_wt_used_bytes -= g_wt[lru].bytes;
        fprintf(stderr, "[cache] Evicted key=%016llx  freed=%.1fMB  used=%.0fMB\n",
                (unsigned long long)g_wt[lru].key,
                (double)g_wt[lru].bytes/1e6,
                (double)g_wt_used_bytes/1e6);
        g_wt[lru].valid = 0;
        g_evictions++;
    }
}

/* Insert weight B[K,N] into cache under key. Returns slot pointer or NULL. */
static WtEntry *cache_insert(uint64_t key, const float *B,
                              uint32_t K, uint32_t N, size_t bytes) {
    /* Find free slot */
    int slot = -1;
    for (int i = 0; i < MAX_CACHE; i++) {
        if (!g_wt[i].valid) { slot = i; break; }
    }
    if (slot < 0) {
        /* Evict LRU to make room */
        uint64_t lru_seq = UINT64_MAX; int lru = 0;
        for (int i = 0; i < MAX_CACHE; i++)
            if (g_wt[i].valid && g_wt[i].last_used < lru_seq)
                { lru_seq = g_wt[i].last_used; lru = i; }
        vkDeviceWaitIdle(g.device);
        vkDestroyBuffer(g.device, g_wt[lru].buf, NULL);
        vkFreeMemory(g.device, g_wt[lru].mem, NULL);
        g_wt_used_bytes -= g_wt[lru].bytes;
        g_wt[lru].valid = 0;
        g_evictions++;
        slot = lru;
    }

    /* Ensure VRAM budget */
    cache_evict_lru(bytes);

    /* Allocate device-local buffer */
    WtEntry *e = &g_wt[slot];
    allocDevBuf((VkDeviceSize)bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                &e->buf, &e->mem);
    if (e->mem == VK_NULL_HANDLE) return NULL;

    /* Upload weight (chunked if > stageSize) */
    double t0 = now_ms();
    uploadChunked(e->buf, B, bytes);
    double upload_ms = now_ms() - t0;

    e->key       = key;
    e->bytes     = bytes;
    e->K         = K;
    e->N         = N;
    e->last_used = ++g_wt_seq;
    e->valid     = 1;
    g_wt_used_bytes += bytes;

    fprintf(stderr, "[cache] Stored key=%016llx  [%u×%u]  %.1fMB  upload=%.1fms  "
            "cache=%.0f/%.0fMB\n",
            (unsigned long long)key, K, N,
            (double)bytes/1e6, upload_ms,
            (double)g_wt_used_bytes/1e6, (double)g_wt_budget/1e6);
    return e;
}

/* ────────────────── Descriptor & compute ────────────────── */

static void updateDesc(VkBuffer bA, size_t szA,
                       VkBuffer bB, size_t szB,
                       VkBuffer bC, size_t szC) {
    VkDescriptorBufferInfo bi[3] = {};
    bi[0].buffer=bA; bi[0].range=szA;
    bi[1].buffer=bB; bi[1].range=szB;
    bi[2].buffer=bC; bi[2].range=szC;
    VkWriteDescriptorSet wd[3] = {};
    for (int i=0;i<3;i++) {
        wd[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wd[i].dstSet=g.dSet; wd[i].dstBinding=(uint32_t)i;
        wd[i].descriptorCount=1;
        wd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wd[i].pBufferInfo=&bi[i];
    }
    vkUpdateDescriptorSets(g.device,3,wd,0,NULL);
}

/* Execute compute shader: dst=bufA×bufB, shapes M,N,K */
static void runCompute(uint32_t M, uint32_t N, uint32_t K) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g.cmdBuf, &bi);

    vkCmdBindPipeline(g.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeline);
    vkCmdBindDescriptorSets(g.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g.pipeLayout, 0,1,&g.dSet,0,NULL);
    uint32_t dims[3] = {M,N,K};
    vkCmdPushConstants(g.cmdBuf, g.pipeLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, dims);
    vkCmdDispatch(g.cmdBuf, (N+7)/8, (M+7)/8, 1);

    /* Barrier: compute → transfer (for download) */
    VkBufferMemoryBarrier bar = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bar.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    bar.srcQueueFamilyIndex=bar.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    bar.buffer=g.bufC; bar.size=(VkDeviceSize)M*N*sizeof(float);
    vkCmdPipelineBarrier(g.cmdBuf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,0,NULL,1,&bar,0,NULL);

    /* Download C */
    VkBufferCopy cp = {0, 0, bar.size};
    vkCmdCopyBuffer(g.cmdBuf, g.bufC, g.stageBuf, 1, &cp);
    vkEndCommandBuffer(g.cmdBuf);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount=1; si.pCommandBuffers=&g.cmdBuf;
    vkQueueSubmit(g.queue,1,&si,g.fence);
    vkWaitForFences(g.device,1,&g.fence,VK_TRUE,UINT64_MAX);
    vkResetFences(g.device,1,&g.fence);
}

/* ────────────────── Vulkan init ────────────────── */

static void vk_init(const char *spirvPath) {
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "VkMatmulSrv"; appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&ici,NULL,&g.instance) != VK_SUCCESS) die("vkCreateInstance");

    uint32_t n=0; vkEnumeratePhysicalDevices(g.instance,&n,NULL);
    if (!n) die("No Vulkan devices");
    VkPhysicalDevice *pd = (VkPhysicalDevice*)malloc(n*sizeof(*pd));
    vkEnumeratePhysicalDevices(g.instance,&n,pd);
    g.phys = pd[0];
    for (uint32_t i=0;i<n;i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[i],&p);
        fprintf(stderr,"[vk] Device %u: %s\n",i,p.deviceName);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) g.phys = pd[i];
    }
    free(pd);
    { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(g.phys,&p);
      fprintf(stderr,"[vk] Selected: %s\n",p.deviceName); }

    /* Print available VRAM */
    { VkPhysicalDeviceMemoryProperties mp;
      vkGetPhysicalDeviceMemoryProperties(g.phys,&mp);
      for (uint32_t i=0;i<mp.memoryHeapCount;i++)
          if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
              fprintf(stderr,"[vk] VRAM heap %u: %.0f MB\n",
                      i,(double)mp.memoryHeaps[i].size/1e6); }

    uint32_t nqf=0; vkGetPhysicalDeviceQueueFamilyProperties(g.phys,&nqf,NULL);
    VkQueueFamilyProperties *qp = (VkQueueFamilyProperties*)malloc(nqf*sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys,&nqf,qp);
    g.queueFamily = UINT32_MAX;
    for (uint32_t i=0;i<nqf;i++)
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.queueFamily=i; break; }
    free(qp);
    if (g.queueFamily==UINT32_MAX) die("No compute queue");

    float prio=1.f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=g.queueFamily; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    if (vkCreateDevice(g.phys,&dci,NULL,&g.device)!=VK_SUCCESS) die("vkCreateDevice");
    vkGetDeviceQueue(g.device,g.queueFamily,0,&g.queue);

    /* Shader */
    size_t sz; uint8_t *spv = load_file(spirvPath,&sz);
    if (!spv) die("load shader");
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize=sz; smci.pCode=(const uint32_t*)spv;
    if (vkCreateShaderModule(g.device,&smci,NULL,&g.shaderModule)!=VK_SUCCESS) die("shader");
    free(spv);
    fprintf(stderr,"[vk] Shader: %zu bytes SPIR-V\n",sz);

    /* Descriptor layout: 3 storage buffers */
    VkDescriptorSetLayoutBinding binds[3]={};
    for(int i=0;i<3;i++){binds[i].binding=(uint32_t)i;binds[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount=1;binds[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dsli={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount=3; dsli.pBindings=binds;
    vkCreateDescriptorSetLayout(g.device,&dsli,NULL,&g.dsetLayout);

    VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,12};
    VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount=1; pli.pSetLayouts=&g.dsetLayout;
    pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr;
    vkCreatePipelineLayout(g.device,&pli,NULL,&g.pipeLayout);

    VkPipelineShaderStageCreateInfo ss={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=g.shaderModule; ss.pName="main";
    VkComputePipelineCreateInfo cpci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage=ss; cpci.layout=g.pipeLayout;
    if (vkCreateComputePipelines(g.device,VK_NULL_HANDLE,1,&cpci,NULL,&g.pipeline)!=VK_SUCCESS)
        die("compute pipeline");

    VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex=g.queueFamily;
    cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(g.device,&cpi,NULL,&g.cmdPool);
    VkCommandBufferAllocateInfo cbai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=g.cmdPool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    vkAllocateCommandBuffers(g.device,&cbai,&g.cmdBuf);

    VkDescriptorPoolSize dps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
    VkDescriptorPoolCreateInfo dpci={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    vkCreateDescriptorPool(g.device,&dpci,NULL,&g.dPool);
    VkDescriptorSetAllocateInfo dsai={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=g.dPool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&g.dsetLayout;
    vkAllocateDescriptorSets(g.device,&dsai,&g.dSet);

    VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(g.device,&fi,NULL,&g.fence);

    /* Staging buffer */
    g.stageSize = (size_t)STAGING_MB*1024*1024;
    VkBufferCreateInfo sbi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sbi.size=g.stageSize; sbi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    sbi.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(g.device,&sbi,NULL,&g.stageBuf);
    VkMemoryRequirements smr; vkGetBufferMemoryRequirements(g.device,g.stageBuf,&smr);
    VkMemoryAllocateInfo smai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    smai.allocationSize=smr.size;
    smai.memoryTypeIndex=findMemType(smr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(g.device,&smai,NULL,&g.stageMem)!=VK_SUCCESS)
        die("staging alloc (BAR too small?)");
    vkBindBufferMemory(g.device,g.stageBuf,g.stageMem,0);
    vkMapMemory(g.device,g.stageMem,0,g.stageSize,0,&g.stageMapped);

    g.capA = g.capC = 0;
    g.bufA = g.bufC = VK_NULL_HANDLE;
    g.memA = g.memC = VK_NULL_HANDLE;

    memset(g_wt, 0, sizeof(g_wt));

    fprintf(stderr,"[vk] Init OK — staging=%zuMB  wt_cache_budget=%zuMB\n",
            g.stageSize>>20, g_wt_budget>>20);
}

/* ────────────────── Matmul implementations ────────────────── */

/*
 * do_matmul_with_buffers: run shader with pre-populated bufA, bufB_dev, bufC
 * A must already be in g.bufA (uploaded via staging)
 * B must be in bufB_dev (device-local, cached or just uploaded)
 */
static int run_matmul(uint32_t M, uint32_t N, uint32_t K,
                       VkBuffer bufB_dev, size_t szB, float *C_out) {
    size_t szA = (size_t)M*K*sizeof(float);
    size_t szC = (size_t)M*N*sizeof(float);
    ensureDevBuf(szC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g.bufC, &g.memC, &g.capC);
    updateDesc(g.bufA, szA, bufB_dev, szB, g.bufC, szC);
    runCompute(M, N, K);
    /* runCompute downloads C into staging[0:szC] */
    memcpy(C_out, g.stageMapped, szC);
    return 0;
}

/* Upload A from socket into g.bufA */
static int upload_A(int fd, uint32_t M, uint32_t K) {
    size_t szA = (size_t)M*K*sizeof(float);
    ensureDevBuf(szA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g.bufA, &g.memA, &g.capA);

    /* Receive A directly into staging, then copy to device */
    size_t off = 0;
    while (off < szA) {
        size_t chunk = szA - off;
        if (chunk > g.stageSize) chunk = g.stageSize;
        if (recv_all(fd, g.stageMapped, chunk) != 0) return -1;

        /* Stage → bufA */
        VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(g.cmdBuf,&bi);
        VkBufferCopy cp={0,(VkDeviceSize)off,(VkDeviceSize)chunk};
        vkCmdCopyBuffer(g.cmdBuf,g.stageBuf,g.bufA,1,&cp);
        VkBufferMemoryBarrier bar={VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bar.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        bar.srcQueueFamilyIndex=bar.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        bar.buffer=g.bufA; bar.offset=(VkDeviceSize)off; bar.size=(VkDeviceSize)chunk;
        vkCmdPipelineBarrier(g.cmdBuf,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,0,NULL,1,&bar,0,NULL);
        vkEndCommandBuffer(g.cmdBuf);
        VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount=1; si.pCommandBuffers=&g.cmdBuf;
        vkQueueSubmit(g.queue,1,&si,g.fence);
        vkWaitForFences(g.device,1,&g.fence,VK_TRUE,UINT64_MAX);
        vkResetFences(g.device,1,&g.fence);
        off += chunk;
    }
    return 0;
}

/* Upload B from socket into a new/existing cache entry */
static WtEntry *upload_B_and_cache(int fd, uint64_t key, uint32_t K, uint32_t N) {
    size_t szB = (size_t)K*N*sizeof(float);
    float *Btmp = (float*)malloc(szB);
    if (!Btmp) return NULL;
    if (recv_all(fd, Btmp, szB) != 0) { free(Btmp); return NULL; }
    WtEntry *e = cache_insert(key, Btmp, K, N, szB);
    free(Btmp);
    return e;
}

/* ─────────────────── Fast single-submit path ────────────────── */
/*
 * compute_cached_fast():
 *   Used when the weight is already in VRAM (a_only=1) AND
 *   szA + szC ≤ stageSize (always true for generation tokens).
 *
 *   ONE command buffer:
 *     stageBuf[0:szA]  → bufA          (upload A from socket)
 *     dispatch shader                   (A × B_cached → C)
 *     bufC → stageBuf[szA:szA+szC]     (download C)
 *   ONE vkQueueSubmit + ONE vkWaitForFences
 *
 *   Eliminates the second fence-wait of the v2 slow path (~43ms → ~22ms).
 *
 *   Falls back to slow path (upload_A + run_matmul, 2 submits) if the
 *   activation + result don't fit in a single staging pass.
 */
static int compute_cached_fast(int fd, uint32_t M, uint32_t N, uint32_t K,
                                WtEntry *e, float *C_out) {
    size_t szA = (size_t)M * K * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    /* Fast path: A + C fit in staging together */
    if (szA + szC <= g.stageSize) {
        double t_recv = now_ms();

        /* Receive A directly into staging[0:szA] */
        if (recv_all(fd, g.stageMapped, szA) != 0) return -1;
        double recv_ms = now_ms() - t_recv;

        /* Ensure device-local A and C buffers are large enough */
        ensureDevBuf(szA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g.bufA, &g.memA, &g.capA);
        ensureDevBuf(szC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &g.bufC, &g.memC, &g.capC);

        /* Update descriptors: A=bufA, B=e->buf (cached weight), C=bufC */
        updateDesc(g.bufA, szA, e->buf, e->bytes, g.bufC, szC);

        double t_gpu = now_ms();

        /* Build ONE command buffer: copy A → dispatch → copy C */
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(g.cmdBuf, &bi);

        /* 1) staging[0:szA] → bufA */
        VkBufferCopy cpA = {0, 0, (VkDeviceSize)szA};
        vkCmdCopyBuffer(g.cmdBuf, g.stageBuf, g.bufA, 1, &cpA);

        /* Barrier: transfer write → shader read on bufA */
        VkBufferMemoryBarrier barA = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barA.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        barA.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        barA.srcQueueFamilyIndex = barA.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barA.buffer = g.bufA; barA.size = (VkDeviceSize)szA;
        vkCmdPipelineBarrier(g.cmdBuf,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 1, &barA, 0, NULL);

        /* 2) Dispatch compute shader */
        vkCmdBindPipeline(g.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeline);
        vkCmdBindDescriptorSets(g.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                g.pipeLayout, 0, 1, &g.dSet, 0, NULL);
        uint32_t dims[3] = {M, N, K};
        vkCmdPushConstants(g.cmdBuf, g.pipeLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, dims);
        vkCmdDispatch(g.cmdBuf, (N + 7) / 8, (M + 7) / 8, 1);

        /* Barrier: shader write → transfer read on bufC */
        VkBufferMemoryBarrier barC = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barC.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barC.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        barC.srcQueueFamilyIndex = barC.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barC.buffer = g.bufC; barC.size = (VkDeviceSize)szC;
        vkCmdPipelineBarrier(g.cmdBuf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &barC, 0, NULL);

        /* 3) bufC → staging[szA:szA+szC] */
        VkBufferCopy cpC = {0, (VkDeviceSize)szA, (VkDeviceSize)szC};
        vkCmdCopyBuffer(g.cmdBuf, g.bufC, g.stageBuf, 1, &cpC);
        vkEndCommandBuffer(g.cmdBuf);

        /* ONE submit, ONE fence wait (vs two in slow path) */
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &g.cmdBuf;
        vkQueueSubmit(g.queue, 1, &si, g.fence);
        vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(g.device, 1, &g.fence);

        double gpu_ms = now_ms() - t_gpu;

        /* Copy result out of staging */
        memcpy(C_out, (uint8_t *)g.stageMapped + szA, szC);

        fprintf(stderr, "[fast] [%u×%u×%u]  recv=%.1fms  gpu=%.1fms  total=%.1fms\n",
                M, N, K, recv_ms, gpu_ms, recv_ms + gpu_ms);
        return 0;
    }

    /* Slow fallback: szA+szC > stageSize (very large prefill batch) */
    if (upload_A(fd, M, K) != 0) return -1;
    return run_matmul(M, N, K, e->buf, e->bytes, C_out);
}

/* ────────────────── Client handler ────────────────── */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t M, N, K;
    uint32_t type;      /* 0=plain, 1=cached */
    uint32_t a_only;    /* 1=don't send B (expect cache hit) */
    uint32_t key_lo;
    uint32_t key_hi;
} ReqHdr;   /* 32 bytes */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t status;
    uint32_t M, N;
} RspHdr;   /* 16 bytes */

static void send_rsp(int fd, uint32_t status, uint32_t M, uint32_t N) {
    RspHdr r = {MAGIC_VK, status, M, N};
    send_all(fd, &r, sizeof(r));
}

static void handle_client(int fd) {
    fprintf(stderr,"[server] Client connected fd=%d\n",fd);
    uint64_t req_n = 0;

    for (;;) {
        ReqHdr req;
        if (recv_all(fd, &req, sizeof(req)) != 0) break;
        if (req.magic != MAGIC_VK) { fprintf(stderr,"[server] bad magic\n"); break; }

        uint32_t M=req.M, N=req.N, K=req.K;
        uint64_t key = ((uint64_t)req.key_hi << 32) | req.key_lo;
        req_n++;

        double t0 = now_ms();
        size_t szC = (size_t)M*N*sizeof(float);
        float *C = (float*)malloc(szC);
        if (!C) { send_rsp(fd,3,0,0); break; }

        if (req.type == TYPE_PLAIN) {
            /* ── Plain matmul: receive A+B, compute, return C ── */
            if (upload_A(fd, M, K) != 0) { free(C); break; }

            size_t szB = (size_t)K*N*sizeof(float);
            float *Btmp = (float*)malloc(szB);
            if (!Btmp) { free(C); send_rsp(fd,3,0,0); break; }
            if (recv_all(fd, Btmp, szB) != 0) { free(Btmp); free(C); break; }

            /* Allocate temp device buffer for B */
            VkBuffer tmpBuf; VkDeviceMemory tmpMem;
            allocDevBuf((VkDeviceSize)szB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &tmpBuf, &tmpMem);
            uploadChunked(tmpBuf, Btmp, szB);
            free(Btmp);

            run_matmul(M, N, K, tmpBuf, szB, C);
            vkDeviceWaitIdle(g.device);
            vkDestroyBuffer(g.device, tmpBuf, NULL);
            vkFreeMemory(g.device, tmpMem, NULL);

            send_rsp(fd, STATUS_OK, M, N);
            send_all(fd, C, szC);

        } else {
            /* ── Cached matmul ── */
            WtEntry *e = cache_lookup(key);

            if (req.a_only && !e) {
                /* Client thought weight was cached, but it was evicted */
                fprintf(stderr,"[cache] STALE key=%016llx → NEED_WEIGHT\n",
                        (unsigned long long)key);
                send_rsp(fd, STATUS_NEED_W, M, N);
                free(C);
                g_cache_misses++;
                continue;
            }

            if (!e) {
                /* ── Cache miss: upload A + B, run, cache B ── */
                if (upload_A(fd, M, K) != 0) { free(C); break; }
                e = upload_B_and_cache(fd, key, K, N);
                if (!e) { free(C); send_rsp(fd,3,0,0); break; }
                g_cache_misses++;
                /* Two-submit slow path (first time for this weight) */
                if (run_matmul(M, N, K, e->buf, e->bytes, C) != 0) { free(C); break; }

            } else if (req.a_only) {
                /* ── Cache hit, a_only=1: FAST single-submit path ── */
                e->last_used = ++g_wt_seq;
                g_cache_hits++;
                if (compute_cached_fast(fd, M, N, K, e, C) != 0) { free(C); break; }

            } else {
                /* ── Cache hit, a_only=0: client sent both A and B.
                 * This path should rarely trigger when using _next_key()
                 * monotonic IDs, but we keep it as a safety net for any
                 * client that still uses address-based keys.
                 * Upload A normally, drain B (discard it — weight is valid
                 * in cache), then run matmul with two submits. ── */
                if (upload_A(fd, M, K) != 0) { free(C); break; }
                {
                    size_t drain = (size_t)K * N * sizeof(float);
                    size_t done  = 0;
                    int    ok    = 1;
                    while (done < drain && ok) {
                        size_t chunk = drain - done;
                        if (chunk > g.stageSize) chunk = g.stageSize;
                        if (recv_all(fd, g.stageMapped, chunk) != 0) ok = 0;
                        done += chunk;
                    }
                    if (!ok) { free(C); break; }
                    fprintf(stderr, "[cache] key=%016llx hit(a_only=0): drained %.0fMB\n",
                            (unsigned long long)key, (double)drain/1e6);
                }
                e->last_used = ++g_wt_seq;
                g_cache_hits++;
                if (run_matmul(M, N, K, e->buf, e->bytes, C) != 0) { free(C); break; }
            }

            send_rsp(fd, STATUS_OK, M, N);
            send_all(fd, C, szC);
        }

        free(C);
        double elapsed = now_ms()-t0;

        /* Periodic stats */
        if (req_n % 50 == 0) {
            fprintf(stderr,"[server] req#%llu  hits=%llu miss=%llu evict=%llu  "
                    "cache=%.0f/%.0fMB\n",
                    (unsigned long long)req_n,
                    (unsigned long long)g_cache_hits,
                    (unsigned long long)g_cache_misses,
                    (unsigned long long)g_evictions,
                    (double)g_wt_used_bytes/1e6,
                    (double)g_wt_budget/1e6);
        }
        (void)elapsed;
    }

    close(fd);
    fprintf(stderr,"[server] Client disconnected after %llu reqs  "
            "hits=%llu miss=%llu  cache=%.0fMB used\n",
            (unsigned long long)req_n,
            (unsigned long long)g_cache_hits,
            (unsigned long long)g_cache_misses,
            (double)g_wt_used_bytes/1e6);
}

/* ────────────────── main ────────────────── */

int main(int argc, char **argv) {
    int port = 8097;
    const char *spirvPath = "matmul.spv";
    if (argc > 1) port = atoi(argv[1]);
    if (argc > 2) spirvPath = argv[2];

    const char *cache_mb = getenv("VK_CACHE_MB");
    if (cache_mb) g_wt_budget = (size_t)atoi(cache_mb)*1024*1024;

    fprintf(stderr,"=== Vulkan Matmul Server v3 (Single-Submit Fast Path) ===\n");
    fprintf(stderr,"Port=%d  SPIR-V=%s  cache_budget=%zuMB\n",
            port, spirvPath, g_wt_budget>>20);

    vk_init(spirvPath);

    int srv = socket(AF_INET,SOCK_STREAM,0); if(srv<0) die("socket");
    int opt=1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr={};
    addr.sin_family=AF_INET; addr.sin_addr.s_addr=INADDR_ANY;
    addr.sin_port=htons((uint16_t)port);
    if (bind(srv,(struct sockaddr*)&addr,sizeof(addr))<0) die("bind");
    if (listen(srv,4)<0) die("listen");
    fprintf(stderr,"[server] Listening on :%d\n",port);

    for (;;) {
        struct sockaddr_in cli; socklen_t cl=sizeof(cli);
        int fd = accept(srv,(struct sockaddr*)&cli,&cl);
        if (fd<0) { perror("accept"); continue; }
        handle_client(fd);
    }
}
