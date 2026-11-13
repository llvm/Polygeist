#!/bin/bash
# Run the standalone proxy-kernel extraction suite through cgeist and the
# affine-to-linalg pipeline.
set +e

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${POLYGEIST_PROXY_KERNEL_OUT:-/tmp/proxy_kernel_extractions_mlir}
CGEIST_BIN=${CGEIST_BIN:-$REPO_ROOT/build/bin/cgeist}
POLYGEIST_OPT_BIN=${POLYGEIST_OPT_BIN:-$REPO_ROOT/build/bin/polygeist-opt}
SRC=$REPO_ROOT/issues/proxy_kernel_extractions/proxy_kernel_extractions.c

if [ -n "${POLYGEIST_CLANG_RESOURCE_DIR:-}" ]; then
  RESOURCE_DIR=$POLYGEIST_CLANG_RESOURCE_DIR
elif [ -d "$REPO_ROOT/llvm-project/build/lib/clang/18" ]; then
  RESOURCE_DIR=$REPO_ROOT/llvm-project/build/lib/clang/18
else
  RESOURCE_DIR=/usr/lib/clang/14
fi

mkdir -p "$OUT"
rm -f "$OUT"/*

count_pattern() {
  local pattern=$1
  local file=$2
  if [ ! -s "$file" ]; then
    echo 0
    return
  fi
  grep -Ec "$pattern" "$file" 2>/dev/null
}

pick_artifact() {
  local tag=$1
  if [ -s "$OUT/${tag}_debuf_mr.mlir" ] &&
     grep -q "linalg.generic" "$OUT/${tag}_debuf_mr.mlir"; then
    echo "$OUT/${tag}_debuf_mr.mlir"
  elif [ -s "$OUT/${tag}_debuf.mlir" ] &&
       grep -q "linalg.generic" "$OUT/${tag}_debuf.mlir"; then
    echo "$OUT/${tag}_debuf.mlir"
  elif [ -s "$OUT/${tag}_linalg.mlir" ]; then
    echo "$OUT/${tag}_linalg.mlir"
  else
    echo "$OUT/${tag}.mlir"
  fi
}

summarize_one() {
  local tag=$1
  local status artifact lg tensor memref loops ifs

  if [ ! -s "$OUT/${tag}.mlir" ]; then
    printf "%-42s %-20s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "cgeist-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.cgeist.err"
    return
  fi
  if [ ! -s "$OUT/${tag}_linalg.mlir" ]; then
    printf "%-42s %-20s %7s %7s %7s %7s %7s %s\n" \
      "$tag" "raise-fail" "-" "-" "-" "-" "-" "$OUT/${tag}.raise.err"
    return
  fi

  artifact=$(pick_artifact "$tag")
  lg=$(count_pattern "linalg\\.generic" "$artifact")
  tensor=$(count_pattern "tensor<" "$artifact")
  memref=$(count_pattern "memref<" "$artifact")
  loops=$(count_pattern "affine\\.for|scf\\.for|affine\\.parallel|scf\\.parallel" "$artifact")
  ifs=$(count_pattern "affine\\.if|scf\\.if" "$artifact")

  if [ "$lg" -gt 0 ] && [ "$tensor" -gt 0 ]; then
    status="tensor-linalg"
  elif [ "$lg" -gt 0 ]; then
    status="memref-linalg"
  else
    status="no-linalg"
  fi
  if [ "$loops" -gt 0 ]; then
    status="${status}+loops"
  fi
  if [ "$ifs" -gt 0 ]; then
    status="${status}+if"
  fi

  printf "%-42s %-20s %7s %7s %7s %7s %7s %s\n" \
    "$tag" "$status" "$lg" "$tensor" "$memref" "$loops" "$ifs" "$artifact"
}

run_probe() {
  local fn=$1
  echo "[$fn] cgeist"
  timeout 90 "$CGEIST_BIN" "$SRC" --function="$fn" \
    --resource-dir="$RESOURCE_DIR" --raise-scf-to-affine -fPIC -std=gnu11 -S \
    -o "$OUT/${fn}.mlir" 2>"$OUT/${fn}.cgeist.err"
  if [ ! -s "$OUT/${fn}.mlir" ]; then
    echo "  cgeist FAILED"
    rm -f "$OUT/${fn}.mlir"
    summarize_one "$fn" >> "$SUMMARY"
    return
  fi

  echo "[$fn] raise"
  timeout 90 "$POLYGEIST_OPT_BIN" \
    --select-func=func-name="$fn" \
    --remove-iter-args --affine-parallelize \
    --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
    "$OUT/${fn}.mlir" -o "$OUT/${fn}_linalg.mlir" \
    2>"$OUT/${fn}.raise.err"
  if [ ! -s "$OUT/${fn}_linalg.mlir" ]; then
    echo "  raise FAILED"
    rm -f "$OUT/${fn}_linalg.mlir"
    summarize_one "$fn" >> "$SUMMARY"
    return
  fi

  echo "[$fn] debuf v2"
  timeout 90 "$POLYGEIST_OPT_BIN" --linalg-debufferize \
    "$OUT/${fn}_linalg.mlir" -o "$OUT/${fn}_debuf.mlir" \
    2>"$OUT/${fn}.debuf.err"
  if [ ! -s "$OUT/${fn}_debuf.mlir" ]; then
    rm -f "$OUT/${fn}_debuf.mlir"
  fi

  echo "[$fn] debuf multi-root"
  timeout 90 "$POLYGEIST_OPT_BIN" --linalg-debufferize=use-multi-root=true \
    "$OUT/${fn}_linalg.mlir" -o "$OUT/${fn}_debuf_mr.mlir" \
    2>"$OUT/${fn}.debuf_mr.err"
  if [ ! -s "$OUT/${fn}_debuf_mr.mlir" ]; then
    rm -f "$OUT/${fn}_debuf_mr.mlir"
  fi

  summarize_one "$fn" >> "$SUMMARY"
}

PROBES=(
  miniamr_stencil_calc_7
  miniamr_stencil_calc_27
  miniamr_stencil_0_coupled_sum
  miniamr_stencil_0_pointwise_update
  miniamr_stencil_x_directional
  miniamr_stencil_y_directional
  miniamr_stencil_z_directional
  miniamr_stencil_7_weighted
  miniamr_stencil_27_weighted
  miniamr_pack_face_x
  miniamr_unpack_face_x
  miniamr_pack_block
  miniamr_unpack_block
  hpgmg_apply_op_7pt
  hpgmg_apply_op_27pt
  hpgmg_residual_7pt
  hpgmg_jacobi_smooth_7pt
  hpgmg_gsrb_smooth_7pt
  hpgmg_zero_vector
  hpgmg_init_vector
  hpgmg_add_vectors
  hpgmg_mul_vectors
  hpgmg_invert_vector
  hpgmg_scale_vector
  hpgmg_shift_vector
  hpgmg_dot
  hpgmg_norm
  hpgmg_mean
  hpgmg_error_l2
  hpgmg_color_vector
  hpgmg_random_vector
  hpgmg_restriction_cell
  hpgmg_restriction_face_i
  hpgmg_restriction_face_j
  hpgmg_restriction_face_k
  hpgmg_interpolation_p0
  hpgmg_interpolation_p1
  hpgmg_interpolation_p2
  hpgmg_fv2_flux
  hpgmg_fv4_flux
  hpgmg_cg_update
  hpgmg_bicgstab_update
  hypar_first_derivative_first_order
  hypar_first_derivative_second_order
  hypar_first_derivative_fourth_order
  hypar_interp_first_order_upwind
  hypar_interp_second_order_central
  hypar_interp_second_order_muscl
  hypar_interp_fourth_order_central
  hypar_interp_fifth_order_weno
  hypar_weno_weights_js
  hypar_limiter_minmod
  hypar_limiter_superbee
  hypar_limiter_generalized_minmod
  hypar_limiter_vanleer
  hypar_linear_adr_advection_const
  hypar_linear_adr_advection_var
  hypar_linear_adr_diffusion_g
  hypar_linear_adr_diffusion_h
  hypar_linear_adr_upwind_const
  hypar_linear_adr_upwind_var
  hypar_linear_adr_centered_flux
  hypar_linear_adr_reaction
  hypar_burgers_advection
  hypar_burgers_upwind
  hypar_euler1d_flux
  hypar_euler1d_llf
  hypar_euler2d_flux_x
  hypar_euler2d_flux_y
  hypar_euler2d_llf
  swfft_redistribute_2_to_3_pack
  swfft_redistribute_3_to_2_unpack
  swfft_slab_pack
  swfft_slab_unpack
  swfft_transpose_xy
  swfft_transpose_yz
  exasp2_normalize_dense
  exasp2_normalize_dense_split
  exasp2_dense_square
  exasp2_sp2_update_2x_minus_x2
  exasp2_sp2_select_square
  exasp2_trace
  exasp2_axpby
  exasp2_spmv
  exasp2_conjugate_gradient_step
)

SUMMARY=$OUT/summary.txt
printf "%-42s %-20s %7s %7s %7s %7s %7s %s\n" \
  "kernel" "status" "linalg" "tensor" "memref" "loops" "ifs" "artifact" > "$SUMMARY"

for fn in "${PROBES[@]}"; do
  run_probe "$fn"
done

echo "Done. Output in $OUT"
cat "$SUMMARY"
