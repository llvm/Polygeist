#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT_DIR="${1:?usage: audit_pva_matcher_candidates.sh AUDIT_DIR}"
PYTHON="${PYTHON:-/usr/bin/python3}"
TIMEOUT="${PVA_MATCH_TIMEOUT:-30}"
MATCHER="$REPO_ROOT/scripts/correctness/kernel_match_rewrite.py"

printf 'source,function,linalg_bodies,selected_matches,candidates,status,symbols\n' \
  >"$AUDIT_DIR/matcher_summary.csv"

tail -n +2 "$AUDIT_DIR/summary.csv" | while IFS=, read -r source function \
    frontend raise debufferize linalg affine scf submap tensor notes; do
  stem="${function#fixture_}"
  report="$AUDIT_DIR/$stem.matcher.log"
  status=pass
  if [[ "$debufferize" != pass || "${tensor:-0}" == 0 ]]; then
    status=no_tensor_linalg
    : >"$report"
  elif ! timeout "${TIMEOUT}s" "$PYTHON" "$MATCHER" \
      "$AUDIT_DIR/$stem.tensor.mlir" --dry-run --show-candidates \
      --show-semantic-only --enable-structured-rewrite \
      >"$AUDIT_DIR/$stem.matcher.stdout" 2>"$report"; then
    status=matcher_failure
  fi
  selected=$(grep -c '^  match ' "$report" || true)
  candidates=$(grep -c '^  kernel_candidate ' "$report" || true)
  symbols=$(sed -nE 's/^  match +body#[^]]*] +([^ ]+).*/\1/p' "$report" | paste -sd+ -)
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$source" "$function" \
    "${tensor:-0}" "$selected" "$candidates" "$status" "$symbols" \
    >>"$AUDIT_DIR/matcher_summary.csv"
done

cat "$AUDIT_DIR/matcher_summary.csv"
