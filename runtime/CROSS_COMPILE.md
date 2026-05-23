# Cross-compiling for Jetson Orin (aarch64 + CUDA) from this x86_64 VM

## Goal

Take a kernel.launch-matched MLIR module, lower it through Phase-2 ABI
(`--lower-kernel-launch-to-cublas`) here on the x86_64 dev VM, and produce an
aarch64 ELF binary that:

1. Calls `polygeist_cublas_dgemm` (our runtime shim).
2. Calls into `libcublas.so` / `libcudart.so` on the target Jetson at runtime.

The Jetson does *not* need Polygeist, MLIR, or `nvcc` — only the CUDA runtime
libs that JetPack already ships at `/usr/local/cuda/lib64`.

## What was installed on this VM (2026-05-23)

| Package | Version | Purpose | Disk |
|---|---|---|---|
| `gcc-aarch64-linux-gnu` | 11.4.0 (Ubuntu 22.04) | aarch64 C cross-compiler + libc sysroot at `/usr/aarch64-linux-gnu/` | ~50 MB |
| `g++-aarch64-linux-gnu` | 11.4.0 | aarch64 C++ cross-compiler (mostly for consistency; we don't use C++ in the shim) | included |
| `binutils-aarch64-linux-gnu` | 2.38 | `ld`, `as`, `readelf` for aarch64 | included |
| `libc6-dev-arm64-cross` | latest | aarch64 libc headers + static libs | included |
| **CUDA cross-sbsa toolkit, 12.6** | 12.6.4.1 | aarch64 (SBSA-ABI) headers + link-time stub libs for `cudart` + `cuBLAS`. Installs to `/usr/local/cuda-12.6/targets/sbsa-linux/{include,lib}`. | ~850 MB |
| └ `cuda-cudart-cross-sbsa-12-6` | 12.6.77 | `cudaMalloc`, `cudaMemcpy`, `cudaFree`, … | (part of above) |
| └ `libcublas-cross-sbsa-12-6` | 12.6.4.1 | `cublasDgemm`, `cublasCreate`, … | (part of above) |
| └ `cuda-nvcc-cross-sbsa-12-6` | 12.6.77 | NOT used to compile — installed only because `cuda_runtime_api.h` `#include`s `crt/host_config.h` which lives in this package | (part of above) |
| └ `cuda-driver-cross-sbsa-12-6` | 12.6.77 | Pulled in transitively; we don't call the driver API directly | (part of above) |
| └ `cuda-cccl-cross-sbsa-12-6` | 12.6.77 | Pulled in transitively (CUDA C++ Core Libraries — unused for us) | (part of above) |

**Total disk footprint:** ~911 MB (`/usr/aarch64-linux-gnu` + `/usr/local/cuda-12.6`).

### Why SBSA and not L4T?

NVIDIA distributes two aarch64 CUDA flavours:

- **L4T (Linux for Tegra)** — what JetPack installs on the Jetson itself.
  No standalone cross-compile apt repo; normally set up via SDK Manager.
- **SBSA (Server Base System Architecture)** — datacenter aarch64
  (Grace, Hopper, etc.). NVIDIA ships a clean apt repo for x86 → SBSA
  cross-compile at
  `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-sbsa/`.

The cuBLAS + cuRT *API surface* and ABI are identical between L4T and SBSA
at runtime — both are 64-bit ARM Linux, same calling convention, same library
layout. So a binary cross-built against SBSA stubs and shipped to a Jetson
will resolve its `libcublas.so.12` / `libcudart.so.12` against JetPack's L4T
copies at load time and work correctly.

### Why also install `gcc-aarch64-linux-gnu` if Polygeist's clang already targets aarch64?

Polygeist's clang knows the aarch64 ISA, but doesn't ship a sysroot (libc,
crt files, libgcc). Using `aarch64-linux-gnu-gcc` as the driver is the
simpler path — it picks up Ubuntu's cross sysroot at `/usr/aarch64-linux-gnu`
automatically. The build scripts below use gcc as the driver for C files and
only invoke clang to compile the `.ll` produced by `mlir-translate`.

### Adding the NVIDIA repo (what was done)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-sbsa/ /' \
  | sudo tee /etc/apt/sources.list.d/cuda-cross-sbsa.list

sudo apt update
sudo apt install -y --no-install-recommends \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu libc6-dev-arm64-cross \
    cuda-cudart-cross-sbsa-12-6 \
    libcublas-cross-sbsa-12-6 \
    cuda-nvcc-cross-sbsa-12-6   # ← needed for crt/host_*.h headers
```

(`shim-signed` may fail to configure during install — that's a UEFI
bootloader package unrelated to CUDA; ignore the dpkg error.)

## How to cross-compile a kernel binary

The end-to-end recipe lives in `scripts/correctness/build_jetson.sh` (with a
local-build variant in `scripts/correctness/gemm_cublas_e2e.sh`). The key
flags:

```bash
# 1. Lower MLIR to LLVM IR (host-side, this VM)
mlir-opt --one-shot-bufferize=bufferize-function-boundaries \
         --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
         --convert-arith-to-llvm --finalize-memref-to-llvm \
         --convert-func-to-llvm --reconcile-unrealized-casts \
         gemm_abi.mlir -o gemm_llvm.mlir
mlir-translate --mlir-to-llvmir gemm_llvm.mlir -o gemm.ll

# 2. Rewrite the .ll's target triple from x86 → aarch64-linux-gnu
sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|' gemm.ll
sed -i '/^target datalayout/d' gemm.ll  # let clang re-derive it for aarch64

# 3. Compile the .ll for aarch64 (clang's aarch64 backend)
CUDA=/usr/local/cuda-12.6/targets/sbsa-linux
clang --target=aarch64-linux-gnu \
      --gcc-toolchain=/usr \
      -O3 -c gemm.ll -o gemm_kernel.o

# 4. Cross-compile the runtime shim
aarch64-linux-gnu-gcc -O3 -c \
    -I$CUDA/include \
    runtime/polygeist_cublas_rt_cuda.c \
    -o polygeist_cublas_rt.o

# 5. Link everything against the aarch64 cuBLAS / cudart stubs
aarch64-linux-gnu-gcc -O2 \
    gemm_kernel.o polygeist_cublas_rt.o <harness>.o \
    -L$CUDA/lib -L$CUDA/lib/stubs \
    -lcublas -lcudart -lm \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -o gemm_jetson
```

The resulting binary:

- ELF 64-bit, ARM aarch64.
- `DT_NEEDED`: `libcublas.so.12`, `libcudart.so.12`, `libc.so.6`,
  `ld-linux-aarch64.so.1`.
- `RUNPATH`: `/usr/local/cuda/lib64` (matches the Jetson's JetPack layout).

scp to the Jetson, `chmod +x`, run — no additional Polygeist or MLIR install
needed on the target.

## Smoke tests done (`/tmp/cross_smoke/`)

| Test | What it proves |
|---|---|
| `hello_aarch64` (gcc) | aarch64 sysroot + binutils work end-to-end |
| `hello_clang_aarch64` | Clang's aarch64 backend + `--gcc-toolchain=/usr` work |
| `tiny_cuda2_aarch64` | Cross-link against `libcudart.so` stub succeeds |
| `tiny_cublas_aarch64` | Cross-link against `libcublas.so` stub succeeds |
| `tiny_polygeist_aarch64` | Our actual `polygeist_cublas_rt_cuda.c` cross-compiles cleanly and links into a tiny driver that calls `polygeist_cublas_dgemm` |

All produce ELF aarch64 binaries with the expected `DT_NEEDED` and
`RUNPATH=/usr/local/cuda/lib64`. None can be executed on the x86 VM (wrong
arch); they're for deployment to the Jetson.

## What's *not* on this VM (and doesn't need to be)

- `nvcc` (host) — we never compile `.cu` files.
- libcublas / libcudart for x86_64 — we don't run CUDA locally; the CPU
  stub at `runtime/polygeist_cublas_rt_cpu.c` covers local validation.
- A working CUDA driver — needed at runtime on the Jetson, not at build
  time on this VM.
- L4T-specific cross-compile env — SBSA is a strict superset of what
  JetPack ships at the BLAS/RT API surface, so we don't need it.

## Updating to a different CUDA version

If the Jetson is on a different CUDA major (e.g. 11.4 from JetPack 5.x, or
12.x where x ≠ 6), `apt install` the matching `*-cross-sbsa-XX-Y` packages
and update the `CUDA=` line in the build script. The cross-sbsa repo has
11-7 through 12-9 currently.
