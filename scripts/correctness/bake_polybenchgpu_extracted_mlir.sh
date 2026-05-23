#!/bin/bash
# Bake the polybenchGpu-extracted kernels (currently conv2d, conv3d) into
# the IR viewer's naming convention:
#   /tmp/pbgpu_extracted_mlir/<tag>.mlir          (post-cgeist affine MLIR)
#   /tmp/pbgpu_extracted_mlir/<tag>_linalg.mlir   (after raise + lower-submap)
#   /tmp/pbgpu_extracted_mlir/<tag>_debuf.mlir    (v2 debufferize)
#   /tmp/pbgpu_extracted_mlir/<tag>_debuf_mr.mlir (multi-root debuf)
#
# These kernels were extracted from the original polybenchGpu/OpenMP .c
# files so that cgeist doesn't inline main→init→kernel and constant-fold
# the conv body away. Each .c here has ONLY the kernel function, with
# A/B as explicit parameters and sizes baked in via #define. The lift
# produces clean linalg.generic ops with ins(A) outs(B). See the
# directory's conv2d.c docstring for the longer explanation.
set +e
source /home/arjaiswal/Polygeist/envsetup.sh
DIR=/home/arjaiswal/Polygeist/third_party/polybenchGpu-extracted
OUT=/tmp/pbgpu_extracted_mlir
mkdir -p $OUT

# Format: <tag>  <fn>  <source_file>
KERNELS=(
  "conv2d  kernel_conv2d  conv2d.c"
  "conv3d  kernel_conv2d  conv3d.c"
)

for entry in "${KERNELS[@]}"; do
  read tag fn srcname <<<"$entry"
  src="$DIR/$srcname"
  [ ! -f "$src" ] && { echo "$tag: missing $src"; continue; }

  echo "[$tag] cgeist..."
  timeout 60 cgeist "$src" --function=$fn --resource-dir=/usr/lib/clang/14 \
      --raise-scf-to-affine -fPIC -S -o $OUT/${tag}.mlir 2>$OUT/${tag}.cgeist.err
  [ ! -s $OUT/${tag}.mlir ] && { echo "  cgeist FAILED"; rm -f $OUT/${tag}.mlir; continue; }

  echo "[$tag] raise..."
  timeout 60 polygeist-opt --select-func=func-name=$fn \
      --remove-iter-args --affine-parallelize \
      --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
      $OUT/${tag}.mlir -o $OUT/${tag}_linalg.mlir 2>$OUT/${tag}.raise.err
  [ ! -s $OUT/${tag}_linalg.mlir ] && { echo "  raise FAILED"; rm -f $OUT/${tag}_linalg.mlir; continue; }

  echo "[$tag] debuf v2..."
  timeout 60 polygeist-opt --linalg-debufferize \
      $OUT/${tag}_linalg.mlir -o $OUT/${tag}_debuf.mlir 2>$OUT/${tag}.debuf.err
  [ ! -s $OUT/${tag}_debuf.mlir ] && { rm -f $OUT/${tag}_debuf.mlir; }

  echo "[$tag] debuf multi-root..."
  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
      $OUT/${tag}_linalg.mlir -o $OUT/${tag}_debuf_mr.mlir 2>$OUT/${tag}.debuf_mr.err
  if [ ! -s $OUT/${tag}_debuf_mr.mlir ]; then
    echo "// Multi-root --linalg-debufferize FAILED. See ${tag}.debuf_mr.err." > $OUT/${tag}_debuf_mr.mlir
  fi
done

echo "Done. Output in $OUT/"
ls $OUT/ | head -20
