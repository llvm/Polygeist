// RUN: env POLYBENCH_DATASET=SMALL POLYBENCH_COMPILE_OPT=-O3 POLYBENCH_E2E_OUTPUT_ROOT=/tmp/polybench-config-test bash -n %S/../../scripts/correctness/run_kernel_e2e.sh
// RUN: grep -F 'POLYBENCH_DATASET:-MINI' %S/../../scripts/correctness/run_kernel_e2e.sh
// RUN: grep -F 'POLYBENCH_COMPILE_OPT:--O1' %S/../../scripts/correctness/run_kernel_e2e.sh
// RUN: grep -F 'POLYBENCH_E2E_OUTPUT_ROOT:-/tmp' %S/../../scripts/correctness/run_kernel_e2e.sh
// RUN: grep -F 'MLIR_OPT:-$REPO_ROOT/llvm-project/build/bin/mlir-opt' %S/../../scripts/correctness/polygeist_build.sh
// RUN: grep -F '[ "$arg" = "-Dstatic=" ] || POLYBENCH_CFLAGS+=' %S/../../scripts/correctness/polygeist_build.sh
// RUN: grep -F '$CC "${GCC_PASSTHROUGH[@]}" -O3 -fno-inline' %S/../../scripts/correctness/polygeist_build.sh
// RUN: grep -F -- '-fsemantic-interposition' %S/../../scripts/correctness/polygeist_build.sh
// RUN: grep -F '[ "$arg" = "-Dstatic=" ] && HARNESS_EXTRA_CFLAGS+=(-fgnu89-inline)' %S/../../scripts/correctness/polygeist_build.sh

// The Section 4.2 evaluator requires one driver to support both cheap smoke
// tests and the paper's explicit LARGE/-O3 protocol, while retaining artifacts
// outside the shared /tmp/e2e_* namespace. The harness must remain uninlined
// so its weak source kernel cannot be cloned around the matched wrapper.
