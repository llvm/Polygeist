# Reproduction commands

The primary fresh build used this command from the repository root.  The harness copy retained beside this file is equivalent to the original temporary path.

```sh
POLYGEIST_BUILD_WORK_DIR=/tmp/polybench-direct-dgemm-rerun-20260908/2mm/correctness-work \
POLYGEIST_KEEP_WORK=1 \
POLYGEIST_MINIMAL_CUDA_RUNTIME=1 \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
PYTHON=/usr/bin/python3 \
scripts/correctness/polygeist_build.sh \
  --target=jetson \
  --function=kernel_2mm \
  --harness=/tmp/polybench-publication-orin/2mm/raised_harness.c \
  --output=/tmp/polybench-direct-dgemm-rerun-20260908/2mm/raised-correctness \
  tools/cgeist/Test/polybench/linear-algebra/kernels/2mm/2mm.c \
  -- -O3 -DCORRECTNESS_RUN \
  -Itools/cgeist/Test/polybench/utilities \
  -Itools/cgeist/Test/polybench/linear-algebra/kernels/2mm \
  -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO
```

The executable was transported and run through the local `pva-general` profile.  The full stdout is retained as `raised-correctness.silicon.log.gz`; repository policy excludes the private endpoint and credentials.  Comparison result:

```text
FAIL values=960000 failures=960000 max_abs=266862.87 max_rel=1 rtol=0.0005 atol=0.011
first_failure index=0 reference=419.21 candidate=0.0
```
