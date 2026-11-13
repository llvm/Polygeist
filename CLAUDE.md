# Polygeist - Claude Instructions

## Environment Setup

Source this before running any commands:
```bash
export POLYGEIST_ROOT=/path/to/Polygeist
source "$POLYGEIST_ROOT/envsetup.sh"
```
This adds `build/bin/` to PATH, making `cgeist` and `polygeist-opt` available.

## Build

Only `build_polygeist.sh` is needed (LLVM/MLIR/Clang are pre-built in `llvm-project/build`).

To rebuild after making changes to any pass:
```bash
cd "$POLYGEIST_ROOT/build" && ninja
```

## Raising Pipeline (C → Linalg)

```bash
# Step 1: C to affine MLIR
cgeist <file.c> --function=* --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -fPIC -S -g -c -o output.mlir

# Step 2: Affine → Linalg (memref form)
polygeist-opt --select-func="func-name=<funcname>" --remove-iter-args --affine-parallelize --raise-affine-to-linalg-pipeline <input.mlir> -o <output_linalg.mlir>

# Step 3: Debufferize (memref linalg → tensor linalg)
polygeist-opt --linalg-debufferize <input_linalg.mlir> -o <output_debufferized.mlir>

# Step 4: Kernel extraction
polygeist-opt <input_debufferized.mlir> --linalg-to-kernel="kernel-library-path=$POLYGEIST_ROOT/generic_solver/kernel_library.mlir"
```

## Key Source Files

- `lib/polygeist/Passes/RaiseToLinalg.cpp` — raises `affine.for` loops to `linalg.generic`, creates `polygeist.submap` for strided accesses
- `lib/polygeist/Passes/LinalgDebufferize.cpp` — converts memref-based linalg to tensor-based SSA form
- `include/polygeist/PolygeistOps.td` — defines `polygeist.submap` and `polygeist.submapInverse`

## NVIDIA gated-distribution SDKs — point, don't copy

The directory `$PVASOL_ROOT` is the source tree for the PVA
Solutions SDK. The PVA Solutions public `.deb` packages ship binaries only
(`libpva_operator.so`, `libnvcv_types.so`, allowlist file) — *no headers*.
Headers exist only inside the source tree, which NVIDIA distributes to
approved developers through `developer.nvidia.com/embedded/pva`. The headers
are therefore "behind a developer-program gate," not "secret internal-only";
they're the same files any approved external developer would have.

*Rule for using these headers in Polygeist:*

- *Build-time include path is fine.* Add `-I$PVASOL_ROOT/public/src/operator/include`
  (and the same pattern for NVCV / cuPVA / CV-CUDA headers under `public/3rdparty/`)
  to the cross-compile flags in our build scripts.
- *Never copy headers into the Polygeist tree.* No `cp` / `git add` of any
  `.h` / `.hpp` / `.cpp` / `.c` from `$PVASOL_ROOT` into
  `$POLYGEIST_ROOT`. The Polygeist repo only ever references those
  paths symbolically.
- *Polygeist source code may `#include "OpConv2d.h"` etc.* — the include is
  resolved through the `-I` flag at build time, just like cuDNN's `cudnn.h`.
- *Anyone cloning Polygeist without PVA Solutions access gets a clean build
  failure* — same as the cuDNN dependency on the cross-compile path today.
- *Same policy applies* to any other gated-distribution NVIDIA SDK source
  tree on this VM (cuPVA SDK, internal NVCV builds, etc.).
