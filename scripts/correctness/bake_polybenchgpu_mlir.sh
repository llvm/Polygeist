#!/bin/bash
# Bake polybenchGpu (OpenMP variant) per-kernel MLIR files in the naming
# convention the IR viewer expects:
#   /tmp/pbgpu_mlir/<tag>.mlir          (post-cgeist affine MLIR)
#   /tmp/pbgpu_mlir/<tag>_linalg.mlir   (after raise + lower-submap)
#   /tmp/pbgpu_mlir/<tag>_debuf.mlir    (default v2 debufferize)
#   /tmp/pbgpu_mlir/<tag>_debuf_mr.mlir (multi-root debufferize)
set +e
source /home/arjaiswal/Polygeist/envsetup.sh
ROOT=/home/arjaiswal/Polygeist/third_party/polybenchGpu/OpenMP
UTIL=$ROOT/utilities
OUT=/tmp/pbgpu_mlir
mkdir -p $OUT

# Format: <tag>  <subdir-under-OpenMP>  <fn>
KERNELS=(
  "correlation        datamining/correlation                     kernel_correlation"
  "covariance         datamining/covariance                      kernel_covariance"
  "2mm                linear-algebra/kernels/2mm                 kernel_2mm"
  "3mm                linear-algebra/kernels/3mm                 kernel_3mm"
  "atax               linear-algebra/kernels/atax                kernel_atax"
  "bicg               linear-algebra/kernels/bicg                kernel_bicg"
  "cholesky           linear-algebra/kernels/cholesky            kernel_cholesky"
  "doitgen            linear-algebra/kernels/doitgen             kernel_doitgen"
  "gemm               linear-algebra/kernels/gemm                kernel_gemm"
  "gemver             linear-algebra/kernels/gemver              kernel_gemver"
  "gesummv            linear-algebra/kernels/gesummv             kernel_gesummv"
  "mvt                linear-algebra/kernels/mvt                 kernel_mvt"
  "symm               linear-algebra/kernels/symm                kernel_symm"
  "syr2k              linear-algebra/kernels/syr2k               kernel_syr2k"
  "syrk               linear-algebra/kernels/syrk                kernel_syrk"
  "trisolv            linear-algebra/kernels/trisolv             kernel_trisolv"
  "trmm               linear-algebra/kernels/trmm                kernel_trmm"
  "durbin             linear-algebra/solvers/durbin              kernel_durbin"
  "dynprog            linear-algebra/solvers/dynprog             kernel_dynprog"
  "gramschmidt        linear-algebra/solvers/gramschmidt         kernel_gramschmidt"
  "lu                 linear-algebra/solvers/lu                  kernel_lu"
  "ludcmp             linear-algebra/solvers/ludcmp              kernel_ludcmp"
  "floyd-warshall     medley/floyd-warshall                      kernel_floyd_warshall"
  "reg_detect         medley/reg_detect                          kernel_reg_detect"
  "adi                stencils/adi                               kernel_adi"
  "convolution-2d     stencils/convolution-2d                    kernel_conv2d"
  "convolution-3d     stencils/convolution-3d                    kernel_conv2d"
  "fdtd-2d            stencils/fdtd-2d                           kernel_fdtd_2d"
  "fdtd-apml          stencils/fdtd-apml                         kernel_fdtd_apml"
  "jacobi-1d-imper    stencils/jacobi-1d-imper                   kernel_jacobi_1d_imper"
  "jacobi-2d-imper    stencils/jacobi-2d-imper                   kernel_jacobi_2d_imper"
  "seidel-2d          stencils/seidel-2d                         kernel_seidel_2d"
)

for entry in "${KERNELS[@]}"; do
  read tag subdir fn <<<"$entry"
  D=$ROOT/$subdir
  src=$(ls $D/*.c 2>/dev/null | head -1)
  [ -z "$src" ] && { echo "$tag: missing source in $D"; continue; }

  # NOTE: polybenchGpu files contain BOTH the kernel and main(); cgeist
  # inlines the kernel into main and DCEs the standalone definition. So
  # we use --function=* and drop --select-func so the raise pass sees the
  # affine loops inside main (where the kernel now lives).
  echo "[$tag] cgeist..."
  timeout 60 cgeist "$src" '--function=*' --resource-dir=/usr/lib/clang/14 \
    -I$UTIL -I$D --raise-scf-to-affine -fPIC -S \
    -o $OUT/${tag}.mlir 2>$OUT/${tag}.cgeist.err
  if [ ! -s $OUT/${tag}.mlir ]; then
    echo "  cgeist FAILED"; rm -f $OUT/${tag}.mlir; continue
  fi

  echo "[$tag] raise..."
  timeout 60 polygeist-opt \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    $OUT/${tag}.mlir -o $OUT/${tag}_linalg.mlir 2>$OUT/${tag}.raise.err
  [ ! -s $OUT/${tag}_linalg.mlir ] && { echo "  raise FAILED"; rm -f $OUT/${tag}_linalg.mlir; continue; }

  echo "[$tag] debuf v2..."
  timeout 60 polygeist-opt --linalg-debufferize \
    $OUT/${tag}_linalg.mlir -o $OUT/${tag}_debuf.mlir 2>$OUT/${tag}.debuf.err
  [ ! -s $OUT/${tag}_debuf.mlir ] && { echo "  v2 debuf FAILED"; rm -f $OUT/${tag}_debuf.mlir; }

  echo "[$tag] debuf multi-root..."
  timeout 60 polygeist-opt --linalg-debufferize=use-multi-root=true \
    $OUT/${tag}_linalg.mlir -o $OUT/${tag}_debuf_mr.mlir 2>$OUT/${tag}.debuf_mr.err
  if [ ! -s $OUT/${tag}_debuf_mr.mlir ]; then
    echo "// Multi-root --linalg-debufferize FAILED. See ${tag}.debuf_mr.err." > $OUT/${tag}_debuf_mr.mlir
  fi
done

echo "Done. Output in $OUT/"
ls $OUT/ | head -30
