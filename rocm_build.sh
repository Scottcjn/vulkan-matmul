#!/bin/bash
# rocm_build.sh — Build ROCm matmul server
# ==========================================
# Requires: ROCm 5.x+ with HIP and rocBLAS installed
#
# ROCm install: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html
# On Ubuntu: sudo apt install rocm-hip-sdk rocblas-dev
#
# Usage:
#   ./rocm_build.sh [gfx_arch]
#   ./rocm_build.sh gfx1012    # Navi 12 (RX 5500M) — offload compute only
#   ./rocm_build.sh gfx1030    # Navi 21 (RX 6800 XT)
#   ./rocm_build.sh gfx1100    # Navi 31 (RX 7900 XTX)
#   ./rocm_build.sh gfx906     # Vega 20 (Radeon VII / MI50)
#   ./rocm_build.sh gfx900     # Vega 10 (RX Vega 64)
#
# The server binary runs on the HOST with the AMD GPU.
# The client (ggml-gpu-offload-rocm.h) runs on the INFERENCE machine.

set -e

# Detect ROCm path
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [ ! -d "$ROCM_PATH" ]; then
    echo "ROCm not found at $ROCM_PATH"
    echo "Install ROCm or set ROCM_PATH"
    exit 1
fi

HIPCC="${ROCM_PATH}/bin/hipcc"
if [ ! -f "$HIPCC" ]; then
    echo "hipcc not found at $HIPCC"
    exit 1
fi

# GPU target architecture
GFX_ARCH="${1:-gfx1010}"  # Default: Navi 10 family (covers Navi 12 via compat)

# rocBLAS include/lib
ROCBLAS_INC="${ROCM_PATH}/include"
ROCBLAS_LIB="${ROCM_PATH}/lib"

echo "=== Building ROCm Matmul Server ==="
echo "ROCm: $ROCM_PATH"
echo "hipcc: $HIPCC"
echo "Target: $GFX_ARCH"
echo "rocBLAS: $ROCBLAS_LIB"
echo ""

# Build server
"$HIPCC" \
    -O3 \
    -std=c++17 \
    --offload-arch="$GFX_ARCH" \
    -I"$ROCBLAS_INC" \
    -L"$ROCBLAS_LIB" \
    -Wl,-rpath,"$ROCBLAS_LIB" \
    rocm_matmul_server.cpp \
    -lrocblas \
    -lpthread \
    -o rocm_matmul_server

echo ""
echo "=== Build OK → ./rocm_matmul_server ==="
echo ""
echo "Usage:"
echo "  ./rocm_matmul_server [port=8098] [cache_mb=7000]"
echo ""
echo "Then on POWER8 (or wherever you run llama.cpp):"
echo "  env GGML_GPU_OFFLOAD_ROCM=1 \\"
echo "      GGML_ROCM_MATMUL_HOST=<server_ip> \\"
echo "      GGML_ROCM_MATMUL_PORT=8098 \\"
echo "      GGML_ROCM_MIN_M=2 \\"
echo "      llama-cli -m model.gguf -ngl 0 -t 1 -p 'prompt' -n 20"
echo ""
echo "Supported GPU architectures:"
echo "  gfx900   — Vega 10 (Radeon RX Vega 64/56)"
echo "  gfx906   — Vega 20 (Radeon VII, MI50/60)"
echo "  gfx1010  — Navi 10 (RX 5700 XT)"
echo "  gfx1012  — Navi 12 (RX 5500M, OCuLink/eGPU use case)"
echo "  gfx1030  — Navi 21 (RX 6800 XT)"
echo "  gfx1100  — Navi 31 (RX 7900 XTX)"
