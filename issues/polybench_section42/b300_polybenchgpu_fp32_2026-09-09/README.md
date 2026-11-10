# Native PolyBenchGPU FP32/LARGE on B300

Date: 2026-09-09

## Outcome

The complete defensible native PolyBenchGPU FP32 set was run on physical B300
GPU 3. Fourteen of 15 normalized programs pass complete-output correctness.
Gram-Schmidt runs but fails correctness (431,605 of 2,640,000 values), so its
timing is diagnostic and must not be used as a performance result.

The original omission was a runner limitation: the B300 PolyBenchGPU runner
was hard-coded to FP64 and supported only 13 normalization adapters. It now
selects FP32/FP64 explicitly and includes canonical live-output adapters for
SYRK and SYR2K.

## Protocol

- NVIDIA B300 SXM6 AC, compute capability 10.3, physical GPU 3; NVIDIA
  driver 580.95.05 and reported CUDA capability 13.1 on Ubuntu 24.04 x86-64.
- PolyBench/C `LARGE_DATASET`, FP32, canonical source inputs and live outputs.
- PolyBenchGPU source revision
  `5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c`.
- Five same-process warmups followed by five synchronized measurements.
- Headline is the minimum; median, maximum, and IQR are retained.
- `e2e_*` covers adapter allocation, transfers, upstream computation, and free.
- `device_*` uses CUDA events around the upstream computational kernels only.
- Correctness compares the complete live output against an independently built
  canonical PolyBench/C FP32 reference with `rtol=5e-4`, `atol=1.1e-2`.
- GPU clock locking was denied in the container. These are secondary,
  non-fixed-clock results and should be labelled that way in the paper.
- The adapters normalize datatype, dimensions, initialization, integration,
  launch geometry, and observable live outputs. They do not constitute
  compiler matches and are not untouched upstream-source results.

The remote campaign command was:

```sh
POLYBENCHGPU_DTYPE=fp32 \
  /mnt/forge1/arpitj/venv/bin/python \
  /mnt/forge1/arpitj/scripts/run_polybenchgpu_b300.py
```

Raw retained logs and full-output dumps are stored outside the repository at
`~/.config/polygeist-silicon/modal-b300/collected-2026-09-09/polybenchgpu-native-large-fp32/`.

## Coverage boundary

The upstream CUDA tree contains 21 programs. Fifteen have a defensible
canonical PolyBench/C counterpart after the stated normalization. Six are not
paper-comparable:

- `2DCONV` and `3DCONV`: no corresponding benchmark in the pinned canonical
  PolyBench/C Section 4.2 corpus.
- `ADI`: the external CUDA source implements an older, non-equivalent ADI
  formulation.
- `JACOBI1D` and `JACOBI2D`: the external CUDA sources replace the canonical
  second stencil with a copy.
- `LU`: the external CUDA source produces a different factor layout.

SYRK and SYR2K were previously rejected because the external kernels also
update the upper triangle. Their canonical live output is only the lower
triangle, so the new adapters compare and return exactly that lower-triangle
contract while clearly retaining the modified-source label.

`raised_comparison.csv` contains the 12 correctness-passing kernels shared with
the existing Polygeist FP32 campaign. Those ratios are only secondary: the two
sides used different physical B300s (same model) and clocks could not be locked.
Values greater than one in `native_over_raised` mean Polygeist was faster.
