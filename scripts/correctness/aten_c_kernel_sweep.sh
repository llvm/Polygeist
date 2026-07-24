#!/usr/bin/env bash
# Raise and match the complete standalone ATen C extraction corpus.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

SRC_DIR="$REPO_ROOT/issues/aten_c_kernels"
OUT="${ATEN_C_SWEEP_OUT:-$SRC_DIR/results}"
CGEIST="$REPO_ROOT/build/bin/cgeist"
OPT="$REPO_ROOT/build/bin/polygeist-opt"
MATCHER="$SCRIPT_DIR/kernel_match_rewrite.py"
MATCH_PYTHON="${ATEN_C_MATCH_PYTHON:-/usr/bin/python3}"
RESOURCE_DIR="$($REPO_ROOT/llvm-project/build/bin/clang -print-resource-dir)"
mkdir -p "$OUT"

printf 'kernel\tstatus\tlinalg_ops\tresidual_loops\tkernel_launches\tmatched_symbols\n' \
  > "$OUT/summary.tsv"

for src in "$SRC_DIR"/aten_*.c; do
  fn="$(basename "$src" .c)"
  dir="$OUT/$fn"
  mkdir -p "$dir"

  if ! timeout 60 "$CGEIST" "$src" --function="$fn" \
      --resource-dir="$RESOURCE_DIR" --raise-scf-to-affine -S \
      -o "$dir/orig.mlir" 2>"$dir/cgeist.err"; then
    printf '%s\tfrontend_failed\t0\t0\t0\t\n' "$fn" | tee -a "$OUT/summary.tsv"
    continue
  fi

  if ! timeout 60 "$OPT" --select-func="func-name=$fn" \
      --remove-iter-args --affine-parallelize \
      --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
      "$dir/orig.mlir" -o "$dir/raised.mlir" 2>"$dir/raise.err"; then
    printf '%s\traise_failed\t0\t0\t0\t\n' "$fn" | tee -a "$OUT/summary.tsv"
    continue
  fi

  if ! timeout 60 "$OPT" --linalg-debufferize "$dir/raised.mlir" \
      -o "$dir/debuf.mlir" 2>"$dir/debuf.err"; then
    printf '%s\tdebufferize_failed\t0\t0\t0\t\n' "$fn" \
      | tee -a "$OUT/summary.tsv"
    continue
  fi

  if ! timeout 60 "$MATCH_PYTHON" "$MATCHER" "$dir/debuf.mlir" \
      >"$dir/matched.mlir" 2>"$dir/match.err"; then
    printf '%s\tmatch_failed\t0\t0\t0\t\n' "$fn" | tee -a "$OUT/summary.tsv"
    continue
  fi

  # Flat aliases consumed by build_ce_viewer.py. Keep the per-kernel
  # directories above as the authoritative logs/artifacts.
  cp "$dir/orig.mlir" "$OUT/$fn.mlir"
  cp "$dir/raised.mlir" "$OUT/${fn}_linalg.mlir"
  cp "$dir/debuf.mlir" "$OUT/${fn}_debuf.mlir"

  linalg_ops="$(rg -c 'linalg\.(generic|matmul|conv)' "$dir/raised.mlir" || true)"
  residual_loops="$(rg -c '\b(affine|scf)\.(for|parallel|while)\b' \
    "$dir/raised.mlir" || true)"
  launches="$(rg -c 'kernel\.launch ' "$dir/matched.mlir" || true)"
  symbols="$({ rg -o 'kernel\.launch @[A-Za-z0-9_]+' \
      "$dir/matched.mlir" || true; } \
    | sed 's/kernel.launch @//' | sort -u | paste -sd, -)"

  printf '%s\tpass\t%s\t%s\t%s\t%s\n' "$fn" "${linalg_ops:-0}" \
    "${residual_loops:-0}" "${launches:-0}" "$symbols" \
    | tee -a "$OUT/summary.tsv"
done

printf 'Artifacts: %s\n' "$OUT"
