#!/usr/bin/env bash
# build_power8.sh — Build vulkan matmul server on POWER8
# Run this ON the POWER8 after SCP'ing the source files

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Vulkan Matmul Server — POWER8 Build Script ==="
echo "Workdir: $SCRIPT_DIR"

# ── Install build deps (if needed) ──────────────────────────────────
if ! command -v glslangValidator &>/dev/null; then
    echo "[+] Installing glslang-tools..."
    sudo apt-get install -y glslang-tools
fi

if ! dpkg -l libvulkan-dev &>/dev/null 2>&1; then
    echo "[+] Installing libvulkan-dev..."
    sudo apt-get install -y libvulkan-dev
fi

# ── Compile GLSL shader → SPIR-V ────────────────────────────────────
echo "[+] Compiling matmul.comp → matmul.spv"
glslangValidator -V matmul.comp -o matmul.spv
echo "    SPIR-V: $(du -h matmul.spv | cut -f1)"

# ── CMake build ──────────────────────────────────────────────────────
echo "[+] CMake configure..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    .

echo "[+] Building..."
cmake --build build -j16

echo ""
echo "=== Build complete! ==="
echo ""
echo "To run the server:"
echo "  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.ppc64le.json \\"
echo "  ./build/vulkan_matmul_server 8097"
echo ""
echo "To test (from another terminal on POWER8):"
echo "  python3 test_client.py"
