#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_DIR="$REPO_ROOT/test/Inputs/pva_matchers"
OUT_DIR="${1:-/tmp/pva_semantic_fixture_audit}"
CGEIST_BIN="${CGEIST_BIN:-$REPO_ROOT/build/bin/cgeist}"
OPT_BIN="${POLYGEIST_OPT_BIN:-$REPO_ROOT/build/bin/polygeist-opt}"
PHASE_TIMEOUT="${PVA_AUDIT_PHASE_TIMEOUT:-30}"
mkdir -p "$OUT_DIR"

for source in image_processing.c yolo_dl.c; do
  cc -std=c11 -Wall -Wextra -Werror -fsyntax-only "$INPUT_DIR/$source" \
    2>"$OUT_DIR/$source.native.err" || exit 1
done

printf 'source,function,frontend,raise,debufferize,linalg_generic,affine_for,scf_for,submap,tensor_linalg,notes\n' \
  >"$OUT_DIR/summary.csv"

for source in image_processing.c yolo_dl.c; do
  while read -r function; do
    stem="${function#fixture_}"
    frontend=pass
    raise=pass
    debufferize=pass
    notes=''

    if ! timeout "${PHASE_TIMEOUT}s" "$CGEIST_BIN" "$INPUT_DIR/$source" --function="$function" \
        --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -S \
        -o "$OUT_DIR/$stem.frontend.mlir" 2>"$OUT_DIR/$stem.frontend.err"; then
      frontend=fail
      raise=blocked
      debufferize=blocked
      notes=frontend_failure
      : >"$OUT_DIR/$stem.raised.mlir"
      : >"$OUT_DIR/$stem.tensor.mlir"
    elif ! timeout "${PHASE_TIMEOUT}s" "$OPT_BIN" --select-func="func-name=$function" \
        --remove-iter-args --affine-parallelize \
        --raise-affine-to-linalg-pipeline --lower-polygeist-submap \
        "$OUT_DIR/$stem.frontend.mlir" -o "$OUT_DIR/$stem.raised.mlir" \
        2>"$OUT_DIR/$stem.raise.err"; then
      raise=fail
      debufferize=blocked
      notes=raising_failure
      : >"$OUT_DIR/$stem.tensor.mlir"
    elif ! timeout "${PHASE_TIMEOUT}s" "$OPT_BIN" --linalg-debufferize "$OUT_DIR/$stem.raised.mlir" \
        -o "$OUT_DIR/$stem.tensor.mlir" 2>"$OUT_DIR/$stem.debufferize.err"; then
      debufferize=fail
      notes=debufferize_failure
    fi

    linalg=$(grep -c 'linalg.generic' "$OUT_DIR/$stem.raised.mlir" || true)
    affine=$(grep -c 'affine.for' "$OUT_DIR/$stem.raised.mlir" || true)
    scf=$(grep -c 'scf.for' "$OUT_DIR/$stem.raised.mlir" || true)
    submap=$(grep -c 'polygeist.submap' "$OUT_DIR/$stem.raised.mlir" || true)
    tensor=$(grep -c 'linalg.generic' "$OUT_DIR/$stem.tensor.mlir" || true)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$source" "$function" "$frontend" "$raise" "$debufferize" \
      "$linalg" "$affine" "$scf" "$submap" "$tensor" "$notes" \
      >>"$OUT_DIR/summary.csv"
  done < <(sed -nE 's/^(void|int) (fixture_[A-Za-z0-9_]+).*/\2/p' "$INPUT_DIR/$source")
done

cat "$OUT_DIR/summary.csv"
