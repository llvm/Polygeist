# Polygeist Custom CUDA Library

This directory holds custom CUDA kernels that are meant to become backend
targets for kernel-definition matching when no vendor library primitive is a
good fit.

## `stencil3d_7pt`

Files:

- `polygeist_stencil3d_7pt.h`
- `polygeist_stencil3d_7pt.cu`

The exported C ABI provides f64 and f32 launch wrappers over
device-accessible structured pointers:

- `polygeist_custom_stencil3d_7pt_f64`
- `polygeist_custom_stencil3d_7pt_f32`

It also provides flat seven-tensor wrappers used by the current
`kernel.launch` lowering path:

- `polygeist_custom_stencil3d_7pt_flat_f64`
- `polygeist_custom_stencil3d_7pt_flat_f32`

The kernel computes a structured center-plus-six-neighbor 3D stencil with
optional extra input and optional per-cell coefficient:

```text
base = base_center * center + base_extra * extra
inner = coeff_center * center
      + coeff_xm * xm + coeff_xp * xp
      + coeff_ym * ym + coeff_yp * yp
      + coeff_zm * zm + coeff_zp * zp
      + coeff_extra * extra
out = base + (coeff ? coeff[i,j,k] : 1) * inner
```

This one kernel definition can cover:

- MiniAMR 7-point average.
- MiniAMR weighted 7-point stencil.
- HPGMG 7-point apply operator.
- HPGMG 7-point residual.
- HPGMG weighted-Jacobi-style smoother.
- PolyBench-style 3D heat/Jacobi 7-point stencils.

The structured API intentionally uses strides in elements and assumes
`input_center` already points at the logical center cell for `(0,0,0)`. That
lets a future lowering pass halo-backed interiors directly without
materializing a dense filter.

The flat ABI mirrors the current raised Linalg lowering: it receives seven
same-shaped tap tensors plus optional `extra`/`coeff` pointers. The CUDA object
copies those host pointers to device buffers, launches a one-thread-per-cell
kernel, and copies the output back. That is correct for today’s host-pointer
runtime ABI, but it is not yet the final zero-copy/device-resident path.

Example standalone build, when `nvcc` is available:

```text
nvcc -O3 -std=c++17 -c custom_library/cuda/polygeist_stencil3d_7pt.cu \
  -o /tmp/polygeist_stencil3d_7pt.o
```

`polygeist_build.sh` links weak CPU fallback definitions by default. To use
the CUDA object, compile it for the target architecture and pass:

```text
POLYGEIST_CUSTOM_CUDA_OBJ=/tmp/polygeist_stencil3d_7pt.o \
  scripts/correctness/polygeist_build.sh --target=jetson ...
```

Current integration status:

1. `generic_solver/kernel_library_phase2.mlir` declares the custom f64 tensor
   definitions.
2. `kernel_match_rewrite.py` routes MiniAMR average/weighted 7-point tensor
   bodies to those definitions.
3. `LowerKernelLaunchToCuBLAS.cpp` lowers them to
   `polygeist_custom_stencil3d_7pt_flat_f64`.
4. `runtime/polygeist_cublas_rt_{cpu,cuda}.c` provide CPU/weak fallback
   implementations so builds run even without the CUDA object.
