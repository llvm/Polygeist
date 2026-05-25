#!/bin/bash
# bake_extracted_darknet_mlir.sh — emit the per-stage MLIR snapshots the
# IR explorer expects for each polybench-style CNN-block kernel in
# third_party/cnn-extracted/.
#
# For each kernel <K> with extracted source at $EXT/<K>.c we produce:
#   /tmp/extracted_darknet_mlir/<K>.mlir         — cgeist output (affine MLIR)
#   /tmp/extracted_darknet_mlir/<K>_linalg.mlir  — after raise (memref linalg)
#   /tmp/extracted_darknet_mlir/<K>_debuf.mlir   — after debufferize (tensor linalg)
#
# These are exactly the three naming conventions build_kernel_page reads
# (raised / debuf tabs + matcher round-trip via the rewriter).

set -euo pipefail
source /home/arjaiswal/Polygeist/envsetup.sh

EXT=/home/arjaiswal/Polygeist/third_party/cnn-extracted
OUT=/tmp/extracted_darknet_mlir
mkdir -p "$OUT"

# (kernel_name, function_name) pairs
KERNELS=(
  "conv2d_batched              kernel_conv2d_batched"
  "maxpool_batched             kernel_maxpool_batched"
  "batchnorm_batched           kernel_batchnorm_batched"
  "shortcut_batched            kernel_shortcut_batched"
  "conv_bn_relu_batched        kernel_conv_bn_relu_batched"
  "conv_bias_relu_add_batched  kernel_conv_bias_relu_add_batched"
  "gemm_bias_relu              kernel_gemm_bias_relu"
  "ata_gemm                    kernel_ata_gemm"
  "conv1x1_batched             kernel_conv1x1_batched"
)

for line in "${KERNELS[@]}"; do
  read -r K FN <<<"$line"
  echo "[$K]"

  cgeist "$EXT/$K.c" --function="$FN" --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine -fPIC -S -g -c -o "$OUT/$K.mlir" 2>"$OUT/$K.cgeist.err" || {
      echo "  cgeist failed; see $OUT/$K.cgeist.err"; continue;
  }

  polygeist-opt --select-func="func-name=$FN" \
      --remove-iter-args --affine-parallelize \
      --raise-affine-to-linalg-pipeline \
      "$OUT/$K.mlir" -o "$OUT/$K"_linalg.mlir 2>"$OUT/$K.raise.err" || {
      echo "  raise failed; see $OUT/$K.raise.err"; continue;
  }

  polygeist-opt --linalg-debufferize \
      "$OUT/$K"_linalg.mlir -o "$OUT/$K"_debuf.mlir 2>"$OUT/$K.debuf.err" || {
      echo "  debuf failed; see $OUT/$K.debuf.err"; continue;
  }

  N_LG=$(grep -c "linalg.generic" "$OUT/$K"_debuf.mlir || true)
  echo "  OK: $N_LG linalg.generic op(s) in debuf"
done
