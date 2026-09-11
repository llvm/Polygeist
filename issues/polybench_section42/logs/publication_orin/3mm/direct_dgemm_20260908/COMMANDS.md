# Reproduction commands

The primary fresh build used this command from the repository root.  The harness copy retained beside this file is equivalent to the original temporary path.

```sh
POLYGEIST_BUILD_WORK_DIR=/tmp/polybench-direct-dgemm-rerun-20260908/3mm/correctness-work \
POLYGEIST_KEEP_WORK=1 \
POLYGEIST_MINIMAL_CUDA_RUNTIME=1 \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
PYTHON=/usr/bin/python3 \
scripts/correctness/polygeist_build.sh \
  --target=jetson \
  --function=kernel_3mm \
  --harness=/tmp/polybench-publication-orin/3mm/raised_harness.c \
  --output=/tmp/polybench-direct-dgemm-rerun-20260908/3mm/raised-correctness \
  tools/cgeist/Test/polybench/linear-algebra/kernels/3mm/3mm.c \
  -- -O3 -DCORRECTNESS_RUN \
  -Itools/cgeist/Test/polybench/utilities \
  -Itools/cgeist/Test/polybench/linear-algebra/kernels/3mm \
  -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO
```

This build did not produce an executable: the matcher found zero launches and final linkage could not resolve `kernel_3mm_impl`.  The valid stored-IR control was transported and run through the local `pva-general` profile; private endpoint and credential details are excluded by repository policy.  Its comparison result was:

```text
FAIL values=880000 failures=880000 max_abs=136791.32 max_rel=1 rtol=0.0005 atol=0.011
first_failure index=0 reference=237.65 candidate=0.0
```
