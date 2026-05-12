# vulkan-matmul

GPU matmul offload server prototypes for POWER8-oriented inference setups.

This repository contains two server paths:

- a **Vulkan** server for AMD GPUs, with a POWER8-focused build flow
- a **ROCm / rocBLAS** server for HIP-capable AMD GPUs

It also includes POWER8 support headers and small client / benchmark helpers for validating the offload path.

## What Is Here

| File | Purpose |
|------|---------|
| `server.cpp` | Vulkan matmul server |
| `rocm_matmul_server.cpp` | ROCm / rocBLAS matmul server |
| `build_power8.sh` | Vulkan build helper for POWER8 systems |
| `rocm_build.sh` | ROCm build helper for HIP-capable AMD GPUs |
| `test_client.py` | basic client smoke test |
| `vk_bench.py` | benchmark helper for the Vulkan server |
| `powerpc/` | POWER8 integration and support code |
| `BCOS.md` | BCOS certification details |

## Quick Start

### Vulkan path on POWER8

```bash
./build_power8.sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.ppc64le.json ./build/vulkan_matmul_server 8097
python3 test_client.py
```

### ROCm path

```bash
./rocm_build.sh gfx1012
./rocm_matmul_server 8098
```

Then point your inference host at the ROCm server:

```bash
env GGML_GPU_OFFLOAD_ROCM=1 \
    GGML_ROCM_MATMUL_HOST=<server_ip> \
    GGML_ROCM_MATMUL_PORT=8098 \
    GGML_ROCM_MIN_M=2 \
    llama-cli -m model.gguf -ngl 0 -t 1 -p 'prompt' -n 20
```

## Build Notes

- `CMakeLists.txt` is used for the Vulkan server build
- `build_power8.sh` installs the Vulkan-side build dependencies if needed
- `rocm_build.sh` expects ROCm and `hipcc` under `/opt/rocm` unless `ROCM_PATH` is overridden
- `vk_bench.py` assumes the Vulkan server is reachable on port `8097` by default

## Related Docs

- [BCOS.md](BCOS.md)
- [powerpc/README.md](powerpc/README.md)
- [powerpc/PSE_IMPLEMENTATION_LOG.md](powerpc/PSE_IMPLEMENTATION_LOG.md)

## License

See the [contribution guide](CONTRIBUTING.md#license) for contribution licensing terms.
