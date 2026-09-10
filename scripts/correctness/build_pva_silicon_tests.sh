#!/usr/bin/env bash
# Cross-build the direct 18-case ABI matrix and the two source-exact,
# automatically matched PVA executables. Deployment is intentionally separate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:?usage: build_pva_silicon_tests.sh OUT_DIR}"
: "${PVASOL_ROOT:?set PVASOL_ROOT to the staged PVA Solutions source tree}"
: "${CUPVA_SDK_ROOT:?set CUPVA_SDK_ROOT to the build-time CUPVA headers}"
: "${PVA_LIB_STAGE:?set PVA_LIB_STAGE to staged AArch64 PVA libraries}"
CUDA_CROSS="${POLYGEIST_CUDA_CROSS_ROOT:-/usr/local/cuda-12.6/targets/sbsa-linux}"
PVA_TARGET_RPATH="${POLYGEIST_PVA_TARGET_RPATH:-/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia}"
mkdir -p "$OUT_DIR"

OPERATOR_INC="$PVASOL_ROOT/public/src/operator/include"
NVCV_INC="$PVASOL_ROOT/public/3rdparty/cvcuda/src/nvcv/src/include"
COMMON_INCLUDES=(-I"$REPO_ROOT/runtime" -I"$CUDA_CROSS/include"
                 -I"$CUPVA_SDK_ROOT/include" -I"$OPERATOR_INC" -I"$NVCV_INC")

aarch64-linux-gnu-gcc -O2 -Wall -Wextra -Werror -ffunction-sections \
  -fdata-sections "${COMMON_INCLUDES[@]}" \
  -c "$REPO_ROOT/runtime/polygeist_pva_image_rt.c" -o "$OUT_DIR/pva_rt.o"
aarch64-linux-gnu-gcc -O2 -std=gnu11 -Wall -Wextra -Werror \
  -c "$REPO_ROOT/scripts/correctness/pva_six_ops_silicon_harness.c" \
  -o "$OUT_DIR/dtype_matrix.o"
aarch64-linux-gnu-gcc "$OUT_DIR/dtype_matrix.o" "$OUT_DIR/pva_rt.o" \
  -L"$PVA_LIB_STAGE" -L"$CUDA_CROSS/lib" -Wl,--allow-shlib-undefined \
  -lpva_operator -lnvcv_types -lcupva_host -lcudart -lm -lpthread -ldl \
  "-Wl,-rpath,$PVA_TARGET_RPATH" \
  -o "$OUT_DIR/pva_dtype_matrix"

export POLYGEIST_ABI_BACKEND=pva
export POLYGEIST_ONLY_KERNELS=pvaGaussianFilter_3x3_u8
export POLYGEIST_HARNESS_CFLAGS=-DPVA_SMOKE_OP=2
"$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
  --function=fixture_gaussian_filter3x3 \
  "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
  --harness="$REPO_ROOT/scripts/correctness/pva_matched_smoke_harness.c" \
  -o "$OUT_DIR/production_gaussian"

export POLYGEIST_ONLY_KERNELS=pvaMorphologyDilate_3x3_u8
export POLYGEIST_HARNESS_CFLAGS=-DPVA_SMOKE_OP=3
"$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
  --function=fixture_morphology_dilate3x3 \
  "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
  --harness="$REPO_ROOT/scripts/correctness/pva_matched_smoke_harness.c" \
  -o "$OUT_DIR/production_morphology"

export POLYGEIST_ONLY_KERNELS=pvaImageHistogram_256_u8_u32
export POLYGEIST_HARNESS_CFLAGS=-DPVA_SMOKE_OP=5
"$REPO_ROOT/scripts/correctness/polygeist_build.sh" --target=jetson \
  --function=fixture_image_histogram_u8 \
  "$REPO_ROOT/test/Inputs/pva_matchers/image_processing.c" \
  --harness="$REPO_ROOT/scripts/correctness/pva_matched_smoke_harness.c" \
  -o "$OUT_DIR/production_histogram"

printf 'Built: %s\n' "$OUT_DIR/pva_dtype_matrix" \
  "$OUT_DIR/production_gaussian" "$OUT_DIR/production_morphology" \
  "$OUT_DIR/production_histogram"
