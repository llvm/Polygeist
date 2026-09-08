# Publication Orin campaign: GEMM

Status: correctness complete for the first LARGE/FP64 gate; publication timing
is not yet complete or accepted.

All executables were cross-compiled on the x86 host.  The Orin only executed
the resulting AArch64 binaries.  Private connection details are deliberately
excluded from this retained directory.

## Configuration

- Source: `tools/cgeist/Test/polybench/linear-algebra/blas/gemm/gemm.c`
- Dataset/type: `LARGE_DATASET`, `DATA_TYPE_IS_DOUBLE`
- Native CPU: AArch64 GCC 11, `-O3`
- Raised CPU: Cgeist -> Linalg raising -> semantic matcher -> real
  OpenBLAS/CBLAS 0.3.31
- Raised GPU: Cgeist -> Linalg raising -> semantic matcher -> real cuBLAS,
  CUDA 12.6 cross toolkit
- Correctness: complete 1,100,000-value output, `rtol=5e-4`, `atol=1.1e-2`,
  reject non-finite values

## Correctness results

- Raised CPU versus native CPU: PASS, `max_abs=0.01`,
  `max_rel=2.33819678e-05`
- Raised GPU versus native CPU: PASS, `max_abs=0.01`,
  `max_rel=2.33819678e-05`

The three `*.dump.log.gz` files are the complete dump regions extracted from
the raw runner logs.  They each contain one begin marker and one end marker.
Build logs, executable hashes, source/library hashes, and numerical comparison
reports are retained alongside them.  Unsanitized runner logs remain only in
the local `/tmp/polybench-publication-orin/gemm` campaign directory because
they contain private deployment metadata.

## Command templates

Native CPU correctness build:

```
aarch64-linux-gnu-gcc -O3 -I<polybench-utilities> -I<gemm-source-dir> \
  -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS \
  gemm.c polybench.c -lm -o native-cpu-correctness
```

Raised CPU correctness build:

```
POLYGEIST_CPU_BLAS=1 \
POLYGEIST_CPU_BLAS_CFLAGS=-I<openblas-source> \
POLYGEIST_CPU_BLAS_LIBS=<single-thread-aarch64-openblas-static-archive> \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
scripts/correctness/polygeist_build.sh --target=jetson-cpu \
  --whole-program --function=kernel_gemm -o raised-cpu-correctness \
  gemm.c -O3 -I<polybench-utilities> -I<gemm-source-dir> \
  -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS
```

Raised GPU correctness build:

```
POLYGEIST_MINIMAL_CUDA_RUNTIME=1 \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
scripts/correctness/polygeist_build.sh --target=jetson \
  --whole-program --function=kernel_gemm -o raised-gpu-correctness \
  gemm.c -O3 -I<polybench-utilities> -I<gemm-source-dir> \
  -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_DUMP_ARRAYS
```

Execution used the configured `pva-general` silicon profile.  CPU execution
used `taskset -c 0`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.
