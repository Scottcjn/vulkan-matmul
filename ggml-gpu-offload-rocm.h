/*
 * ggml-gpu-offload-rocm.h — ROCm Matmul Offload Hook for llama.cpp
 * ==================================================================
 * Drop-in companion to ggml-gpu-offload-vulkan.h using the ROCm server.
 * Same protocol v2 wire format, just different env vars and default port.
 *
 * Integration (in ggml-cpu.c):
 *   // Line ~56 — choose ONE of Vulkan OR ROCm:
 *   #include "arch/powerpc/ggml-gpu-offload-rocm.h"
 *
 *   // Line ~1240 (inside ggml_compute_forward_mul_mat, before CPU path):
 *   if (rocm_try_offload_mt(params, src0, src1, dst)) return;
 *
 * Environment variables:
 *   GGML_GPU_OFFLOAD_ROCM=1      Enable (default: off)
 *   GGML_ROCM_MATMUL_HOST=...    Server host (default: 127.0.0.1)
 *   GGML_ROCM_MATMUL_PORT=...    Server port (default: 8098)
 *   GGML_ROCM_MAX_MB=...         Max tensor size to offload in MB (default: 200)
 *   GGML_ROCM_MIN_M=...          Min batch M to offload (default: 2)
 *
 * Notes:
 *   - Run with -t 1 (single thread) so params->nth == 1
 *   - Run with -ngl 0 if using CPU backend alongside ROCm server
 *   - For models that fit in VRAM, use llama.cpp's native ROCm backend instead
 */

#ifndef GGML_GPU_OFFLOAD_ROCM_H
#define GGML_GPU_OFFLOAD_ROCM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#ifdef __cplusplus
#include "ggml.h"
extern "C" {
#else
#include "ggml.h"
#endif

/* ── Wire protocol (identical to Vulkan server) ───────────────────────────*/

#define ROCM_MAGIC        0x564B4D54u  /* "VKMT" — same as Vulkan server */
#define ROCM_STATUS_OK    0u
#define ROCM_STATUS_NEED_W 2u

/* ── Config struct ─────────────────────────────────────────────────────── */

typedef struct {
    int    init_done;
    int    enabled;
    char   host[64];
    int    port;
    int    sock_fd;
    size_t max_bytes;
    int    min_m;

    /* Stats */
    int    req_warm;
    int    req_cold;
    int    req_retry;
    int    req_skip;
    double total_ms;
} _rocm_cfg_t;

static _rocm_cfg_t _rocm_cfg;

/* ── Warm-key hash table (4096 slots, open-addressing) ─────────────────── */

#define ROCM_KEY_HT_SIZE 4096
typedef struct { uint64_t key; int valid; } _RocmKeySlot;
static _RocmKeySlot _rocm_ht[ROCM_KEY_HT_SIZE];

static void _rocm_ht_set(uint64_t key) {
    uint32_t idx = (uint32_t)(key ^ (key >> 32)) & (ROCM_KEY_HT_SIZE - 1);
    for (int i = 0; i < ROCM_KEY_HT_SIZE; i++) {
        uint32_t s = (idx + i) & (ROCM_KEY_HT_SIZE - 1);
        if (!_rocm_ht[s].valid || _rocm_ht[s].key == key) {
            _rocm_ht[s].key   = key;
            _rocm_ht[s].valid = 1;
            return;
        }
    }
}

static int _rocm_ht_has(uint64_t key) {
    uint32_t idx = (uint32_t)(key ^ (key >> 32)) & (ROCM_KEY_HT_SIZE - 1);
    for (int i = 0; i < ROCM_KEY_HT_SIZE; i++) {
        uint32_t s = (idx + i) & (ROCM_KEY_HT_SIZE - 1);
        if (!_rocm_ht[s].valid) return 0;
        if (_rocm_ht[s].key == key) return 1;
    }
    return 0;
}

static void _rocm_ht_del(uint64_t key) {
    uint32_t idx = (uint32_t)(key ^ (key >> 32)) & (ROCM_KEY_HT_SIZE - 1);
    for (int i = 0; i < ROCM_KEY_HT_SIZE; i++) {
        uint32_t s = (idx + i) & (ROCM_KEY_HT_SIZE - 1);
        if (!_rocm_ht[s].valid) return;
        if (_rocm_ht[s].key == key) { _rocm_ht[s].valid = 0; return; }
    }
}

/* ── Network helpers ────────────────────────────────────────────────────── */

static int _rocm_send_all(int fd, const void* p, size_t n) {
    const char* q = (const char*)p;
    while (n > 0) {
        ssize_t r = send(fd, q, n, 0);
        if (r <= 0) { close(_rocm_cfg.sock_fd); _rocm_cfg.sock_fd = -1; return -1; }
        q += r; n -= r;
    }
    return 0;
}

static int _rocm_recv_all(int fd, void* p, size_t n) {
    char* q = (char*)p;
    while (n > 0) {
        ssize_t r = recv(fd, q, n, MSG_WAITALL);
        if (r <= 0) { close(_rocm_cfg.sock_fd); _rocm_cfg.sock_fd = -1; return -1; }
        q += r; n -= r;
    }
    return 0;
}

/* ── Connection management ──────────────────────────────────────────────── */

static int _rocm_connected(void) {
    return _rocm_cfg.sock_fd >= 0;
}

static int _rocm_connect(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)_rocm_cfg.port);
    inet_pton(AF_INET, _rocm_cfg.host, &addr.sin_addr);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    int flag = 1;
    setsockopt(fd, 6 /* IPPROTO_TCP */, 1 /* TCP_NODELAY */, &flag, sizeof(flag));

    _rocm_cfg.sock_fd = fd;
    fprintf(stderr, "[rocm-offload] Connected to %s:%d\n",
            _rocm_cfg.host, _rocm_cfg.port);
    return 0;
}

/* ── Init (called once) ─────────────────────────────────────────────────── */

static void _rocm_init_once(void) {
    if (_rocm_cfg.init_done) return;
    _rocm_cfg.init_done = 1;
    _rocm_cfg.sock_fd   = -1;

    const char *en = getenv("GGML_GPU_OFFLOAD_ROCM");
    _rocm_cfg.enabled = en && atoi(en);

    const char *ps = getenv("GGML_ROCM_MATMUL_PORT");
    _rocm_cfg.port = ps ? atoi(ps) : 8098;

    const char *hs = getenv("GGML_ROCM_MATMUL_HOST");
    strncpy(_rocm_cfg.host, hs ? hs : "127.0.0.1", sizeof(_rocm_cfg.host) - 1);

    const char *mb = getenv("GGML_ROCM_MAX_MB");
    _rocm_cfg.max_bytes = (size_t)((mb ? atof(mb) : 200.0) * 1024 * 1024);

    const char *mm = getenv("GGML_ROCM_MIN_M");
    _rocm_cfg.min_m = mm ? atoi(mm) : 2;

    if (_rocm_cfg.enabled)
        fprintf(stderr, "[rocm-offload] Enabled — %s:%d  max=%.0fMB  min_m=%d\n",
                _rocm_cfg.host, _rocm_cfg.port,
                (double)_rocm_cfg.max_bytes / (1024*1024),
                _rocm_cfg.min_m);
}

/* ── Core offload logic ─────────────────────────────────────────────────── */

static int rocm_try_offload(const struct ggml_tensor *src0,
                             const struct ggml_tensor *src1,
                             struct ggml_tensor *dst)
{
    _rocm_init_once();
    if (!_rocm_cfg.enabled) return 0;

    /* Dimension check: C[M,N] = A[M,K] × B[K,N]
     * In ggml: src1=A (activations), src0=B (weights) */
    if (src0->n_dims < 2 || src1->n_dims < 2) return 0;

    int64_t K = src0->ne[0];
    int64_t N = src0->ne[1];
    int64_t M = src1->ne[1];

    if ((int)M < _rocm_cfg.min_m) { _rocm_cfg.req_skip++; return 0; }

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB_f32 = (size_t)K * N * sizeof(float);

    if (szA + szB_f32 > _rocm_cfg.max_bytes) { _rocm_cfg.req_skip++; return 0; }

    /* Get type-specific weight size */
    const struct ggml_type_traits *tt = NULL;
    size_t szB_quant = szB_f32;
    if (src0->type != GGML_TYPE_F32) {
        tt = ggml_get_type_traits(src0->type);
        if (!tt || !tt->to_float) { _rocm_cfg.req_skip++; return 0; }
        szB_quant = ggml_nbytes(src0);
    }

    /* Connect if needed */
    if (!_rocm_connected()) {
        if (_rocm_connect() < 0) { _rocm_cfg.req_skip++; return 0; }
    }

    /* Unique weight key */
    uint64_t key = (uint64_t)(uintptr_t)src0->data;
    int a_only   = _rocm_ht_has(key) ? 1 : 0;

    uint32_t key_lo = (uint32_t)(key & 0xFFFFFFFFu);
    uint32_t key_hi = (uint32_t)(key >> 32);

    /* Determine ggml type code */
    uint32_t type_code;
    switch (src0->type) {
        case GGML_TYPE_F32:  type_code = 0;  break;
        case GGML_TYPE_F16:  type_code = 1;  break;
        case GGML_TYPE_Q8_0: type_code = 8;  break;
        case GGML_TYPE_Q4_K: type_code = 12; break;
        case GGML_TYPE_Q6_K: type_code = 14; break;
        default:
            _rocm_cfg.req_skip++;
            return 0;
    }

retry:;
    /* Send 32-byte header */
    uint32_t hdr[8] = {
        ROCM_MAGIC, (uint32_t)M, (uint32_t)N, (uint32_t)K,
        type_code, (uint32_t)a_only, key_lo, key_hi
    };
    if (_rocm_send_all(_rocm_cfg.sock_fd, hdr, 32) < 0) return 0;

    /* Send A (always FP32) */
    /* A is src1 in ggml, shape [K, M], row-major */
    if (_rocm_send_all(_rocm_cfg.sock_fd, src1->data, szA) < 0) return 0;

    /* Send B (if cold) */
    if (!a_only) {
        if (_rocm_send_all(_rocm_cfg.sock_fd, src0->data, szB_quant) < 0) return 0;
    }

    /* Read 16-byte response header */
    uint32_t rhdr[4];
    if (_rocm_recv_all(_rocm_cfg.sock_fd, rhdr, 16) < 0) return 0;

    if (rhdr[1] == ROCM_STATUS_NEED_W) {
        /* Server evicted — resend with B */
        _rocm_ht_del(key);
        a_only = 0;
        _rocm_cfg.req_retry++;
        goto retry;
    }

    if (rhdr[1] != ROCM_STATUS_OK) return 0;

    /* Read result into dst */
    size_t szC = (size_t)M * N * sizeof(float);
    if (_rocm_recv_all(_rocm_cfg.sock_fd, dst->data, szC) < 0) return 0;

    if (!a_only) _rocm_ht_set(key);

    _rocm_cfg.total_ms += 0.0;  /* TODO: timing */
    if (a_only) _rocm_cfg.req_warm++; else _rocm_cfg.req_cold++;

    int total = _rocm_cfg.req_warm + _rocm_cfg.req_cold;
    if (total % 100 == 0 && total > 0) {
        fprintf(stderr, "[rocm-offload] ops=%d warm=%d cold=%d retry=%d skip=%d\n",
                total, _rocm_cfg.req_warm, _rocm_cfg.req_cold,
                _rocm_cfg.req_retry, _rocm_cfg.req_skip);
    }

    return 1;  /* handled by GPU */
}

/* ── Multi-thread wrapper ───────────────────────────────────────────────── */

/*
 * rocm_try_offload_mt():
 *   Safe wrapper that only fires on thread 0 in a single-thread context.
 *   With -t 1, params->nth == 1 and params->ith == 0 — offload fires.
 *   With -t N>1, falls through to CPU (multi-thread safe).
 */
static int rocm_try_offload_mt(const struct ggml_compute_params *params,
                                const struct ggml_tensor         *src0,
                                const struct ggml_tensor         *src1,
                                struct ggml_tensor               *dst)
{
    if (params->ith != 0 || params->nth != 1) return 0;
    return rocm_try_offload(src0, src1, dst);
}

/* ── Shutdown stats ─────────────────────────────────────────────────────── */

static void rocm_offload_print_stats(void) {
    if (!_rocm_cfg.init_done || !_rocm_cfg.enabled) return;
    int total = _rocm_cfg.req_warm + _rocm_cfg.req_cold;
    fprintf(stderr, "\n[rocm-offload] ── Final Stats ──\n");
    fprintf(stderr, "  Ops: %d  (warm=%d cold=%d retry=%d skipped=%d)\n",
            total, _rocm_cfg.req_warm, _rocm_cfg.req_cold,
            _rocm_cfg.req_retry, _rocm_cfg.req_skip);
    if (_rocm_cfg.sock_fd >= 0) close(_rocm_cfg.sock_fd);
}

#ifdef __cplusplus
}
#endif

#endif /* GGML_GPU_OFFLOAD_ROCM_H */
