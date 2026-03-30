# Contributing to vulkan-matmul

Thanks for contributing to `vulkan-matmul`.

This repository is a focused GPU offload prototype for POWER8 systems, so the most useful contributions are usually narrow improvements to the Vulkan / ROCm server code, build scripts, or supporting documentation.

## Good Contribution Targets

- Fix correctness or stability issues in `server.cpp` or `rocm_matmul_server.cpp`
- Improve POWER8-specific build behavior in `CMakeLists.txt` or `build_power8.sh`
- Improve ROCm build coverage in `rocm_build.sh`
- Tighten Vulkan / ROCm integration notes in `BCOS.md`
- Improve small test or benchmarking helpers like `test_client.py` or `vk_bench.py`

## Before You Open a PR

1. Read `BCOS.md` and the build scripts to understand the current scope
2. Keep the diff small and tied to one clear problem
3. Avoid unrelated refactors across the POWER8 and ROCm paths in the same PR
4. If you change behavior, update the related usage notes or scripts in the same branch

## Development Notes

- `server.cpp` is the Vulkan server entry point
- `rocm_matmul_server.cpp` is the ROCm / rocBLAS server path
- `build_power8.sh` is the quickest reference for Vulkan builds on POWER8
- `rocm_build.sh` is the quickest reference for HIP / rocBLAS builds
- `powerpc/` contains the POWER8-oriented support code and integration headers

## Suggested Validation

Run the lightest checks that match your change:

```bash
git diff --check
python3 -m py_compile test_client.py vk_bench.py
bash -n build_power8.sh rocm_build.sh
cmake -S . -B build
```

If your machine does not have Vulkan, ROCm, or POWER8-specific tooling available, note that clearly in the PR and list the checks you were still able to run.

## Pull Request Checklist

- Fork the repository and create a topic branch
- Explain the problem being fixed or the workflow being improved
- Include the exact validation commands you ran
- Link the related issue or bounty in the PR body when applicable
- Keep generated files and unrelated formatting noise out of the diff

## Documentation Expectations

When a change affects one of these areas, update the matching file in the same PR:

- build flags or configure behavior -> `CMakeLists.txt` or the relevant build script
- runtime usage or port defaults -> `BCOS.md`
- client-side invocation or smoke testing -> `test_client.py`
- POWER8 integration details -> files under `powerpc/`

## License

By contributing, you agree that your contributions will be released under the repository's existing license terms.
