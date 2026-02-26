/*
 * GGML Local GPU Offload (POWER8 + RX 580)
 * Elyan Labs - Model weights in RAM, matmul on local GPU
 * 
 * Protocol: v3 compatible (0x47505533)
 * Server: OpenCL matmul server at localhost:8096
 */

#ifndef GGML_GPU_OFFLOAD_LOCAL_H
#define GGML_GPU_OFFLOAD_LOCAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <time.h>

#define GPU_LOCAL_PORT 8096
#define GPU_LOCAL_HOST "127.0.0.1"
#define GPU_LOCAL_MAGIC 0x47505533
#define GPU_MIN_DIM_FOR_OFFLOAD 256  // Only offload matrices >= this size

static int gpu_local_socket = -1;
static int gpu_local_connected = 0;
static int gpu_local_enabled = 0;
static int gpu_local_req_count = 0;
static double gpu_local_total_time_ms = 0;

static inline double get_time_ms_local(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static inline int gpu_local_init(void) {
    if (gpu_local_connected) return 0;
    
    char *env = getenv("GGML_GPU_OFFLOAD");
    if (!env || strcmp(env, "1") != 0) {
        return -1;  // Not enabled
    }
    
    gpu_local_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (gpu_local_socket < 0) {
        fprintf(stderr, "[gpu-local] socket failed: %s\n", strerror(errno));
        return -1;
    }
    
    // Set TCP_NODELAY for low latency
    int flag = 1;
    setsockopt(gpu_local_socket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
    // Large send/recv buffers for matrix data
    int bufsize = 64 * 1024 * 1024;  // 64MB
    setsockopt(gpu_local_socket, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(gpu_local_socket, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(GPU_LOCAL_PORT);
    inet_pton(AF_INET, GPU_LOCAL_HOST, &addr.sin_addr);
    
    if (connect(gpu_local_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[gpu-local] connect failed: %s\n", strerror(errno));
        close(gpu_local_socket);
        gpu_local_socket = -1;
        return -1;
    }
    
    gpu_local_connected = 1;
    gpu_local_enabled = 1;
    fprintf(stderr, "[gpu-local] Connected to OpenCL matmul server at %s:%d\n", 
            GPU_LOCAL_HOST, GPU_LOCAL_PORT);
    return 0;
}

static inline int gpu_local_recv_all(void *buf, size_t len) {
    size_t received = 0;
    while (received < len) {
        ssize_t n = recv(gpu_local_socket, (char*)buf + received, len - received, 0);
        if (n <= 0) {
            fprintf(stderr, "[gpu-local] recv failed\n");
            return -1;
        }
        received += n;
    }
    return 0;
}

static inline int gpu_local_send_all(const void *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(gpu_local_socket, (const char*)buf + sent, len - sent, 0);
        if (n <= 0) {
            fprintf(stderr, "[gpu-local] send failed\n");
            return -1;
        }
        sent += n;
    }
    return 0;
}

/* 
 * Offload FP32 matmul to local GPU
 * C[M,N] = A[M,K] @ B[K,N]
 * Returns 0 on success, -1 on failure (caller should fall back to CPU)
 */
static inline int gpu_local_matmul_f32(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    // Initialize connection on first call
    if (!gpu_local_connected) {
        if (gpu_local_init() < 0) {
            return -1;
        }
    }
    
    // Skip small matrices (overhead not worth it)
    if (M < GPU_MIN_DIM_FOR_OFFLOAD && N < GPU_MIN_DIM_FOR_OFFLOAD && K < GPU_MIN_DIM_FOR_OFFLOAD) {
        return -1;  // Use CPU
    }
    
    double t0 = get_time_ms_local();
    
    // Send request header
    uint32_t header[6] = {GPU_LOCAL_MAGIC, M, N, K, 0, 0};  // F32 types
    if (gpu_local_send_all(header, 24) < 0) {
        gpu_local_connected = 0;
        return -1;
    }
    
    // Send matrices
    if (gpu_local_send_all(A, M * K * sizeof(float)) < 0 ||
        gpu_local_send_all(B, K * N * sizeof(float)) < 0) {
        gpu_local_connected = 0;
        return -1;
    }
    
    // Receive response header
    uint32_t resp[4];
    if (gpu_local_recv_all(resp, 16) < 0) {
        gpu_local_connected = 0;
        return -1;
    }
    
    if (resp[0] != GPU_LOCAL_MAGIC || resp[1] != 0) {
        fprintf(stderr, "[gpu-local] error: magic=%x status=%u\n", resp[0], resp[1]);
        return -1;
    }
    
    // Receive result
    if (gpu_local_recv_all(C, M * N * sizeof(float)) < 0) {
        gpu_local_connected = 0;
        return -1;
    }
    
    double t1 = get_time_ms_local();
    gpu_local_req_count++;
    gpu_local_total_time_ms += (t1 - t0);
    
    // Log occasionally
    if (gpu_local_req_count % 100 == 0) {
        fprintf(stderr, "[gpu-local] %d matmuls, avg %.1fms, total %.1fs\n",
                gpu_local_req_count, 
                gpu_local_total_time_ms / gpu_local_req_count,
                gpu_local_total_time_ms / 1000.0);
    }
    
    return 0;
}

static inline void gpu_local_close(void) {
    if (gpu_local_socket >= 0) {
        close(gpu_local_socket);
        gpu_local_socket = -1;
        gpu_local_connected = 0;
    }
    if (gpu_local_req_count > 0) {
        fprintf(stderr, "[gpu-local] Session: %d matmuls, avg %.1fms\n",
                gpu_local_req_count, gpu_local_total_time_ms / gpu_local_req_count);
    }
}

#endif // GGML_GPU_OFFLOAD_LOCAL_H
