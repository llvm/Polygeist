# Essential reproduction commands

The repository root was `/home/arjaiswal/Polygeist`.  Private SSH routing is
provided by the configured `pva-general` runner profile and is omitted here.

## Raised FP32 recognition and correctness build

```
POLYGEIST_MINIMAL_CUDA_RUNTIME=1 \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
POLYGEIST_GPU_ARCH=sm_87 \
POLYGEIST_BUILD_WORK_DIR=/tmp/polybench-fp32-gemm/raised-work-submap \
scripts/correctness/polygeist_build.sh --target=jetson \
  --whole-program --function=kernel_gemm \
  -o /tmp/polybench-fp32-gemm/raised-fp32-submap \
  tools/cgeist/Test/polybench/linear-algebra/blas/gemm/gemm.c \
  -O3 \
  -Itools/cgeist/Test/polybench/utilities \
  -Itools/cgeist/Test/polybench/linear-algebra/blas/gemm \
  -DLARGE_DATASET -DDATA_TYPE_IS_FLOAT -DPOLYBENCH_DUMP_ARRAYS

POLYGEIST_SILICON_PROFILE=pva-general \
scripts/correctness/run_jetson.sh \
  --exe /tmp/polybench-fp32-gemm/raised-fp32-submap \
  gemm_raised_fp32_large_correctness
```

## Full-output comparisons

```
python3 scripts/correctness/compare_polybench_dumps.py \
  scripts/correctness/logs/gemm_native_fp32_large_correctness_20260908_150939.silicon.log \
  scripts/correctness/logs/gemm_raised_fp32_large_correctness_20260908_150950.silicon.log \
  --rtol 5e-4 --atol 1.1e-2

python3 scripts/correctness/compare_polybench_dumps.py \
  issues/polybench_section42/logs/publication_orin_cpu/gemm/gemm-native-cpu-correctness_20260908_114908.silicon.log \
  scripts/correctness/logs/gemm_raised_direct_dgemm_correctness_20260908_150738.silicon.log \
  --rtol 5e-4 --atol 1.1e-2
```

Both commands reported `PASS values=1100000 failures=0`.

## Execution of timed binaries

```
POLYGEIST_SILICON_PROFILE=pva-general \
scripts/correctness/run_jetson.sh \
  --exe /tmp/polybench-publication-orin/gemm/direct-dgemm-timing \
  gemm_raised_direct_dgemm_timing_5x5

POLYGEIST_SILICON_PROFILE=pva-general \
scripts/correctness/run_jetson.sh \
  --exe /tmp/polybench-fp32-gemm/raised-fp32-timing \
  gemm_raised_fp32_timing_5x5

POLYGEIST_SILICON_PROFILE=pva-general \
scripts/correctness/run_jetson.sh \
  --exe /tmp/polybench-fp32-gemm/native-fp32-timing-5x5 \
  gemm_native_gpu_fp32_timing_5x5
```

Each timed executable performs five warmups followed by five samples in one
process.  See the retained silicon logs for every raw sample.
