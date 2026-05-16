#!/bin/bash
# Bake polybenchified-NPB per-kernel MLIR files in the naming the IR
# viewer expects:
#   /tmp/npb_mlir/<tag>.mlir          (post-cgeist affine MLIR)
#   /tmp/npb_mlir/<tag>_linalg.mlir   (after raise + lower-submap)
#   /tmp/npb_mlir/<tag>_debuf.mlir    (default v2 debufferize)
#   /tmp/npb_mlir/<tag>_debuf_mr.mlir (multi-root debufferize)
set +e
source /home/arjaiswal/Polygeist/envsetup.sh
DIR=/home/arjaiswal/Polygeist/third_party/NPB-polybenchified
OUT=/tmp/npb_mlir
mkdir -p $OUT

# Format: <tag>  <fn>  <source_file>
KERNELS=(
  "bt-add       bt_add       bt_add.c"
  "ft-evolve    ft_evolve    ft_evolve.c"
  "lu-l2norm    lu_l2norm    lu_l2norm.c"
  "mg-psinv     mg_psinv     mg_psinv.c"
  "mg-resid     mg_resid     mg_resid.c"
  "mg-norm2u3   mg_norm2u3   mg_norm2u3.c"
  "mg-rprj3     mg_rprj3     mg_rprj3.c"
)

for entry in "${KERNELS[@]}"; do
  read tag fn srcname <<<"$entry"
  src="$DIR/$srcname"
  [ ! -f "$src" ] && { echo "$tag: missing $src"; continue; }

  echo "[$tag] cgeist..."
  timeout 60 cgeist "$src" --function=$fn --resource-dir=/usr/lib/clang/14 \
    --raise-scf-to-affine -fPIC -S -o $OUT/${tag}.mlir 2>$OUT/${tag}.cgeist.err
  if [ ! -s $OUT/${tag}.mlir ]; then
    echo "  cgeist FAIL"; rm -f $OUT/${tag}.mlir; continue
  fi

  echo "[$tag] raise..."
  timeout 60 polygeist-opt --select-func=func-name=$fn \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${tag}.mlir -o $OUT/${tag}_linalg.mlir 2>$OUT/${tag}.raise.err
  [ ! -s $OUT/${tag}_linalg.mlir ] && { echo "  raise FAIL"; rm -f $OUT/${tag}_linalg.mlir; continue; }

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
ls $OUT/ | head -30
