/*
 * ggml-gpu-offload-vulkan.h  v2 — VRAM Weight Cache
 *
 * Transparent Navi 12 GPU offload for llama.cpp on IBM POWER8.
 *
 * Architecture:
 *   Model stays 100% in POWER8 512GB RAM.
 *   Per-token (M=1) generation:
 *     Cold call: send A (28KB) + B (196MB) → server caches B in VRAM
 *     Warm calls: send A (28KB) only → ~22ms via single-submit fast path
 *   Prefill (M≥MIN_M):
 *     Always warm after first token → GPU GEMM faster than POWER8 CPU
 *
 * Why this works on POWER8 / Navi 12:
 *   256MB PCIe BAR limit prevents llama.cpp's -ngl (full layer in VRAM).
 *   Individual matmuls (activation A = M×K×4 bytes) always fit.
 *   Weight B is uploaded once then stays in Navi 12's 8GB VRAM.
 *
 * Protocol v2 (32-byte request header):
 *   magic(4) M(4) N(4) K(4) type(4) a_only(4) key_lo(4) key_hi(4)
 *   type=1 (CACHED):
 *     a_only=0 → send A + B (first use — server caches B)
 *     a_only=1 → send A only (server uses cached B) → fast path
 *   Response: magic(4) status(4) M(4) N(4) + C[M*N*4]
 *   status=0 OK | status=2 NEED_WEIGHT (server evicted B, retry with a_only=0)
 *
 * Integration:
 *   1. Start: ./build/vulkan_matmul_server 8097  (on same machine)
 *   2. Env:   export GGML_GPU_OFFLOAD_VULKAN=1
 *   3. In ggml-cpu.c, before ggml_compute_forward_mul_mat CPU path:
 *        if (vk_try_offload(src0, src1, dst)) return;
 *
 * Environment variables:
 *   GGML_GPU_OFFLOAD_VULKAN=1      Enable (default: off)
 *   GGML_VK_MATMUL_PORT=8097      Server port
 *   GGML_VK_MATMUL_HOST=127.0.0.1 Server host
 *   GGML_VK_MIN_M=2               Minimum batch M to offload (default: 2)
 *                                  At M=1 GPU dispatch overhead > CPU savings.
 *                                  At M≥2 the v3 single-submit fast path wins.
 *   GGML_VK_MAX_MB=200            Max combined A+B size in MB (default: 200)
 *
 * Weight key:
 *   Uses (uint64_t)src0->data — the pointer to the weight tensor's data.
 *   Stable for the lifetime of a loaded model (weights are never moved).
 *   No collision risk between different weight tensors.
 */

#pragma once
#ifndef GGML_GPU_OFFLOAD_VULKAN_H
#define GGML_GPU_OFFLOAD_VULKAN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>

#ifndef GGML_TYPE_F32
#  include "ggml.h"
#endif

/* ─── Protocol constants ────────────────────────────────────────────────── */

#define VK_MAGIC       0x564B4D54u  /* "VKMT" */
#define VK_STATUS_OK   0u
#define VK_STATUS_NW   2u           /* NEED_WEIGHT: server evicted key */
#define VK_TYPE_PLAIN  0u
#define VK_TYPE_CACHED 1u

/* ─── Client-side warm-key hash table ──────────────────────────────────── */
/*
 * Tracks which weight keys the server currently has in VRAM.
 * Open-addressing with linear probe, power-of-2 size.
 * On NEED_WEIGHT response the key is evicted from this table.
 */
#define VK_KEY_HT_SIZE 4096   /* must be power of 2; supports ~2800 weights per model */

typedef struct { uint64_t key; int valid; } _VkKeySlot;
static _VkKeySlot _vk_ht[VK_KEY_HT_SIZE];

static unsigned int _vk_ht_hash(uint64_t key) {
    /* Murmur finaliser mix */
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (unsigned int)(h & (VK_KEY_HT_SIZE - 1));
}

static int _vk_key_is_warm(uint64_t key) {
    unsigned int h = _vk_ht_hash(key);
    for (int i = 0; i < 32; i++) {
        int idx = (h + i) & (VK_KEY_HT_SIZE - 1);
        if (!_vk_ht[idx].valid)          return 0;
        if (_vk_ht[idx].key == key)      return 1;
    }
    return 0;
}

static void _vk_key_mark_warm(uint64_t key) {
    unsigned int h = _vk_ht_hash(key);
    for (int i = 0; i < VK_KEY_HT_SIZE; i++) {
        int idx = (h + i) & (VK_KEY_HT_SIZE - 1);
        if (!_vk_ht[idx].valid || _vk_ht[idx].key == key) {
            _vk_ht[idx].key   = key;
            _vk_ht[idx].valid = 1;
            return;
        }
    }
    /* Table full (shouldn't happen with 4096 slots) — evict at home slot */
    _vk_ht[h].key   = key;
    _vk_ht[h].valid = 1;
}

static void _vk_key_mark_cold(uint64_t key) {
    unsigned int h = _vk_ht_hash(key);
    for (int i = 0; i < 32; i++) {
        int idx = (h + i) & (VK_KEY_HT_SIZE - 1);
        if (!_vk_ht[idx].valid) return;
        if (_vk_ht[idx].key == key) { _vk_ht[idx].valid = 0; return; }
    }
}

/* ─── Config ────────────────────────────────────────────────────────────── */

static struct {
    int    enabled;
    int    sock_fd;
    int    port;
    char   host[64];
    size_t max_bytes;    /* max szA + szB bytes to offload */
    int    min_m;        /* skip offload if M < min_m */
    /* Stats */
    int    req_warm;     /* warm-path requests (a_only=1) */
    int    req_cold;     /* cold-path requests (a_only=0, first upload) */
    int    req_retry;    /* NEED_WEIGHT retries */
    int    req_skip;     /* skipped (too small or too large) */
    double total_ms;
    int    init_done;
} _vk_cfg;

static double _vk_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

static int _vk_send_all(const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n > 0) {
        ssize_t r = send(_vk_cfg.sock_fd, p, n, 0);
        if (r <= 0) { close(_vk_cfg.sock_fd); _vk_cfg.sock_fd = -1; return -1; }
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int _vk_recv_all(void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n > 0) {
        ssize_t r = recv(_vk_cfg.sock_fd, p, n, MSG_WAITALL);
        if (r <= 0) { close(_vk_cfg.sock_fd); _vk_cfg.sock_fd = -1; return -1; }
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int _vk_connect(void) {
    if (_vk_cfg.sock_fd >= 0) return 1;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)_vk_cfg.port);
    inet_pton(AF_INET, _vk_cfg.host, &addr.sin_addr);

    struct timeval tv = {0, 50000};   /* 50ms connect timeout */
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd); return 0;
    }

    tv.tv_sec = 30; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    _vk_cfg.sock_fd = fd;
    fprintf(stderr, "[vk-offload] Connected to %s:%d\n", _vk_cfg.host, _vk_cfg.port);
    return 1;
}

static void _vk_init_once(void) {
    if (_vk_cfg.init_done) return;
    _vk_cfg.init_done = 1;
    _vk_cfg.sock_fd   = -1;

    const char *en = getenv("GGML_GPU_OFFLOAD_VULKAN");
    _vk_cfg.enabled = en && atoi(en);

    const char *ps = getenv("GGML_VK_MATMUL_PORT");
    _vk_cfg.port = ps ? atoi(ps) : 8097;

    const char *hs = getenv("GGML_VK_MATMUL_HOST");
    strncpy(_vk_cfg.host, hs ? hs : "127.0.0.1", sizeof(_vk_cfg.host) - 1);

    const char *mb = getenv("GGML_VK_MAX_MB");
    _vk_cfg.max_bytes = (size_t)((mb ? atof(mb) : 200.0) * 1024 * 1024);

    const char *mm = getenv("GGML_VK_MIN_M");
    _vk_cfg.min_m = mm ? atoi(mm) : 2;  /* GPU wins at M≥2 with fast path */

    if (_vk_cfg.enabled)
        fprintf(stderr,
                "[vk-offload] Enabled — %s:%d  max=%.0fMB  min_m=%d\n",
                _vk_cfg.host, _vk_cfg.port,
                (double)_vk_cfg.max_bytes / (1024*1024),
                _vk_cfg.min_m);
}

/* ─── Core offload function ──────────────────────────────────────────────── */
/*
 * vk_try_offload():
 *   Called before CPU ggml_compute_forward_mul_mat.
 *   dst = src0 × src1  (ggml convention):
 *     src0: weight [K, N]  — may be F32 or any quantized type (Q4_K, Q8_0…)
 *     src1: input  [K, M]  — must be F32 (llama.cpp activation tensors are F32)
 *     dst:  output [N, M]  — F32
 *
 *   Quantized weights are dequantized to F32 on the FIRST upload only.
 *   All subsequent calls for the same weight key send activations only.
 *   This preserves the VRAM caching benefit for quantized models.
 *
 *   Returns 1 if GPU handled it (dst populated), 0 to fall back to CPU.
 */
static int vk_try_offload(const struct ggml_tensor *src0,
                           const struct ggml_tensor *src1,
                           struct ggml_tensor       *dst)
{
    _vk_init_once();
    if (!_vk_cfg.enabled) return 0;

    /* src1 (activations) and dst must be F32 */
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return 0;

    /* Only 2D matmul */
    if (ggml_n_dims(src0) > 2 || ggml_n_dims(src1) > 2) return 0;

    /* Extract dimensions (ggml layout):
         src0: [ne00=K, ne01=N]   weight
         src1: [ne00=K, ne11=M]   activations
         dst:  [ne0=N,  ne1=M]    result      */
    uint32_t K = (uint32_t)src0->ne[0];
    uint32_t N = (uint32_t)src0->ne[1];
    uint32_t M = (uint32_t)src1->ne[1];

    if ((uint32_t)src1->ne[0] != K) return 0;
    if ((uint32_t)dst->ne[0]  != N) return 0;
    if ((uint32_t)dst->ne[1]  != M) return 0;

    size_t szA = (size_t)M * K * 4;   /* activations F32 */
    size_t szB = (size_t)K * N * 4;   /* weights F32 (after optional dequant) */
    size_t szC = (size_t)M * N * 4;   /* result F32 */

    /* Skip if batch too small (GPU dispatch floor > CPU savings) */
    if ((int)M < _vk_cfg.min_m) { _vk_cfg.req_skip++; return 0; }

    /* Skip if too large for staging (A+B combined) */
    if (szA + szB > _vk_cfg.max_bytes) { _vk_cfg.req_skip++; return 0; }

    if (!_vk_connect()) return 0;

    const float *src1_data = (const float *)src1->data;
    float       *dst_data  = (float *)dst->data;

    /* Weight key: stable pointer (weights never move after model load) */
    uint64_t wkey   = (uint64_t)(uintptr_t)src0->data;
    int      a_only = _vk_key_is_warm(wkey) ? 1 : 0;

    /* Dequantize weight to F32 if needed — only on cold upload.
     * For warm hits (a_only=1) we never send B so no dequant needed. */
    float *B_f32 = NULL;
    if (!a_only) {
        if (src0->type == GGML_TYPE_F32) {
            B_f32 = (float *)src0->data;  /* use directly, no alloc */
        } else {
            /* Ask ggml for the dequantization function for this type */
            const struct ggml_type_traits *tt = ggml_get_type_traits(src0->type);
            if (!tt || !tt->to_float) { _vk_cfg.req_skip++; return 0; }
            B_f32 = (float *)malloc(szB);
            if (!B_f32) return 0;
            tt->to_float(src0->data, B_f32, (int64_t)K * N);
        }
    }

    /* Transpose src1 [K,M] → A [M,K] for server convention */
    float *A = (float *)malloc(szA);
    if (!A) { if (B_f32 && src0->type != GGML_TYPE_F32) free(B_f32); return 0; }
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t k = 0; k < K; k++)
            A[m * K + k] = src1_data[k * M + m];

    double t0 = _vk_now_ms();

retry:
    {
        uint32_t hdr[8] = {
            VK_MAGIC, M, N, K,
            VK_TYPE_CACHED, (uint32_t)a_only,
            (uint32_t)(wkey & 0xFFFFFFFF),
            (uint32_t)(wkey >> 32)
        };
        if (_vk_send_all(hdr, 32)  != 0) goto fail;
        if (_vk_send_all(A,   szA) != 0) goto fail;
        if (!a_only) {
            if (_vk_send_all(B_f32, szB) != 0) goto fail;
        }

        uint32_t rhdr[4];
        if (_vk_recv_all(rhdr, 16) != 0) goto fail;
        if (rhdr[0] != VK_MAGIC) {
            fprintf(stderr, "[vk-offload] bad magic 0x%08X\n", rhdr[0]);
            goto fail;
        }
        if (rhdr[1] == VK_STATUS_NW) {
            /* Server evicted weight — re-dequantize and retry with full B */
            _vk_key_mark_cold(wkey);
            if (a_only && B_f32 == NULL) {
                /* Need to dequantize now (was warm, now cold) */
                if (src0->type == GGML_TYPE_F32) {
                    B_f32 = (float *)src0->data;
                } else {
                    const struct ggml_type_traits *tt = ggml_get_type_traits(src0->type);
                    if (!tt || !tt->to_float) goto fail;
                    B_f32 = (float *)malloc(szB);
                    if (!B_f32) goto fail;
                    tt->to_float(src0->data, B_f32, (int64_t)K * N);
                }
            }
            a_only = 0;
            _vk_cfg.req_retry++;
            goto retry;
        }
        if (rhdr[1] != VK_STATUS_OK) {
            fprintf(stderr, "[vk-offload] server error status=%u\n", rhdr[1]);
            goto fail;
        }
        if (_vk_recv_all(dst_data, szC) != 0) goto fail;
    }

    /* Success */
    free(A);
    if (B_f32 && src0->type != GGML_TYPE_F32) free(B_f32);
    _vk_key_mark_warm(wkey);

    {
        double elapsed = _vk_now_ms() - t0;
        _vk_cfg.total_ms += elapsed;
        if (a_only) _vk_cfg.req_warm++; else _vk_cfg.req_cold++;
        int total = _vk_cfg.req_warm + _vk_cfg.req_cold;
        if (total % 100 == 0)
            fprintf(stderr,
                    "[vk-offload] %d ops (warm=%d cold=%d retry=%d skip=%d) "
                    "avg=%.1fms  hit%%=%.0f\n",
                    total, _vk_cfg.req_warm, _vk_cfg.req_cold,
                    _vk_cfg.req_retry, _vk_cfg.req_skip,
                    _vk_cfg.total_ms / total,
                    100.0 * _vk_cfg.req_warm / total);
    }
    return 1;

fail:
    free(A);
    if (B_f32 && src0->type != GGML_TYPE_F32) free(B_f32);
    return 0;
}

/*
 * vk_try_offload_mt():
 *   Multi-thread-safe wrapper.  ggml calls ggml_compute_forward_mul_mat from
 *   N worker threads simultaneously.  We do the GPU dispatch only from
 *   thread 0, and only when running single-threaded (nth==1), which avoids
 *   race conditions without requiring a mutex.
 *
 *   For multi-threaded use (nth>1) this returns 0 and falls through to CPU.
 *   To enable GPU offload in a multi-threaded run, pass -t 1 to llama-cli,
 *   or extend this function with a per-tensor completion flag.
 *
 *   Usage in ggml-cpu.c (inside ggml_compute_forward_mul_mat, before CPU path):
 *     if (vk_try_offload_mt(params, src0, src1, dst)) return;
 */
static int vk_try_offload_mt(const struct ggml_compute_params *params,
                              const struct ggml_tensor         *src0,
                              const struct ggml_tensor         *src1,
                              struct ggml_tensor               *dst)
{
    /* Only thread 0 in a single-threaded context may offload.
     * Multi-threaded matmul (nth>1) falls through to CPU.
     * Use -t 1 with llama-cli to enable GPU offload. */
    if (params->ith != 0 || params->nth != 1) return 0;
    return vk_try_offload(src0, src1, dst);
}

/* Print stats on shutdown */
static void vk_offload_print_stats(void) {
    if (!_vk_cfg.init_done || !_vk_cfg.enabled) return;
    int total = _vk_cfg.req_warm + _vk_cfg.req_cold;
    fprintf(stderr, "\n[vk-offload] ── Final Stats ──\n");
    fprintf(stderr, "  Ops: %d  (warm=%d cold=%d retry=%d skipped=%d)\n",
            total, _vk_cfg.req_warm, _vk_cfg.req_cold,
            _vk_cfg.req_retry, _vk_cfg.req_skip);
    fprintf(stderr, "  Avg ms/op: %.1f  hit%%: %.0f\n",
            total ? _vk_cfg.total_ms / total : 0.0,
            total ? 100.0 * _vk_cfg.req_warm / total : 0.0);
    if (_vk_cfg.sock_fd >= 0) close(_vk_cfg.sock_fd);
}

#endif /* GGML_GPU_OFFLOAD_VULKAN_H */
