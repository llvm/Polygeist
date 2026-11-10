#!/usr/bin/env bash
# build_v2.sh -- C kernel -> lib_v2 library calls -> executable.
#
# Usage:
#   build_v2.sh [--target=host|jetson] [--function=NAME] [--harness=H.c]
#               [-o OUT] [--work=DIR] [--no-region-match] [--no-match]
#               [--no-legaliser] [--matched=MATCHED.mlir]
#               <kernel.c> [cflags...]
#
# Pipeline (steps 1-2 = recipe.py PRODUCTION, steps 5-9 mirror
# polygeist_build.sh's later lowering/link steps, which are not modified):
#   1. raise       cgeist + raise-affine-to-linalg + multi-root debufferize
#   2. match       match_v2.py (lib_v2 Linalg-level matcher + region matcher)
#   3. lower       lower_v2.py: kernel.launch @netlib_* -> polygeist_v2_<call>
#                  (bufferizes with the production pre-ABI flow first)
#   4. runtime     gen_v2_runtime.py --check (runtime wrappers up to date)
#   5. LLVM        production view cleanup + mlir-opt lowering + translate
#   6. wrapper     gen_wrapper.py C-ABI bridge to <fn>_impl
#   7. harness     --harness, or generate_polybench_harness.py (declares, does
#                  not define, the kernel; no symbol interposition)
#   8. compile     host: clang/OpenBLAS CPU runtime (polygeist_v2_rt_cpu.c)
#                  jetson: aarch64 cross + cuBLAS/cuSOLVER (polygeist_v2_rt_cuda.c)
#   9. link
# --matched resumes from an existing match_v2.py output (skips steps 1-2),
# like polygeist_build.sh --semantic-mlir.
# Unmatched code stays as Linalg/affine and is lowered to loops in step 5.
# Views (polygeist.submap) go through the production lowering first; only if
# tensor-form views survive it (dynamic-size reduction write-backs in
# unmatched code) does lower_v2.py fall back to Agent 1's test-only
# legalize_tensor_submaps.py (disable with --no-legaliser).
#
# cflags are passed to cgeist and to every C compile (PolyBench: -I<util>
# -I<kernel dir> '-Dstatic=__attribute__((noipa))' -D<SIZE>_DATASET
# -DDATA_TYPE_IS_DOUBLE -DPOLYBENCH_USE_C99_PROTO ...).
#
# Environment:
#   POLYGEIST_V2_CPU_BLAS_LIBS   host BLAS/LAPACK link flags (default -lopenblas)
#   POLYGEIST_CUDA_CROSS_ROOT    cross CUDA root (default
#                                /usr/local/cuda-12.6/targets/sbsa-linux)
#   POLYGEIST_V2_KEEP_WORK=1     keep the temporary work directory
set -euo pipefail
_V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_V2_DIR/../common_env.sh"

V2=$_V2_DIR
CUDA_CROSS="${POLYGEIST_CUDA_CROSS_ROOT:-/usr/local/cuda-12.6/targets/sbsa-linux}"
AARCH64_CC=aarch64-linux-gnu-gcc

TARGET=host FUNCTION= OUT= INPUT= HARNESS= WORK= REGION=1 MATCH=1
PASS=() LOWER_ARGS=() MATCHED_IN=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --target=*)   TARGET="${1#--target=}"; shift ;;
    --function=*) FUNCTION="${1#--function=}"; shift ;;
    --harness=*)  HARNESS="${1#--harness=}"; shift ;;
    --work=*)     WORK="${1#--work=}"; shift ;;
    --no-region-match) REGION=0; shift ;;
    --no-match)   MATCH=0; shift ;;
    --no-legaliser) LOWER_ARGS+=(--no-legaliser); shift ;;
    --matched=*)  MATCHED_IN="${1#--matched=}"; shift ;;
    -o)           OUT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,32p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *.c) if [ -z "$INPUT" ]; then INPUT="$1"; else PASS+=("$1"); fi; shift ;;
    *)            PASS+=("$1"); shift ;;
  esac
done
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "ERROR: no input .c" >&2; exit 1; }
case "$TARGET" in host|jetson) ;; *) echo "ERROR: --target=host|jetson" >&2; exit 1 ;; esac
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
[ -n "$OUT" ] || OUT="$(basename "$INPUT" .c)_v2"
if [ -z "$FUNCTION" ]; then
  FUNCTION=$(awk '/^(static[ \t]+)?void[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*\(/ {
                    match($0, /void[ \t]+[A-Za-z_][A-Za-z0-9_]*/); s = substr($0, RSTART, RLENGTH);
                    sub(/void[ \t]+/, "", s); last = s }
                  /#pragma[ \t]+scop/ { print last; exit }' "$INPUT")
  [ -n "$FUNCTION" ] || { echo "ERROR: cannot detect kernel; pass --function" >&2; exit 1; }
fi
if [ -z "$WORK" ]; then
  WORK=$(mktemp -d)
  [ "${POLYGEIST_V2_KEEP_WORK:-0}" != 0 ] || trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK"
echo "[v2] input=$INPUT function=$FUNCTION target=$TARGET out=$OUT work=$WORK"

if [ -n "$MATCHED_IN" ]; then
  [ -f "$MATCHED_IN" ] || { echo "ERROR: $MATCHED_IN not found" >&2; exit 1; }
  echo "  [1-2/9] reuse matched MLIR $MATCHED_IN"
  [ "$(cd "$(dirname "$MATCHED_IN")" && pwd)/$(basename "$MATCHED_IN")" = "$WORK/matched.mlir" ] || \
    cp "$MATCHED_IN" "$WORK/matched.mlir"
else
# ─── 1. raise (shared lib_v2 recipe) ────────────────────────────────────
echo "  [1/9] raise (recipe.py PRODUCTION)"
$PYTHON - "$V2" "$INPUT" "$FUNCTION" "$WORK/kernel" "${PASS[@]}" <<'EOF' || exit 1
import sys
sys.path.insert(0, sys.argv[1])
import recipe as R
src, fn, prefix, flags = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
r = R.raise_c(R.PRODUCTION, src, fn, prefix, flags, 900)
for step in ("cgeist", "raise", "debuf"):
    if step not in r or not r[step].ok:
        st = r.get(step)
        print(f"ERROR: {step} failed" + (f": {st.stderr[-2000:]}" if st else ""), file=sys.stderr)
        sys.exit(1)
EOF

# ─── 2. match ───────────────────────────────────────────────────────────
if [ "$MATCH" -eq 1 ]; then
  echo "  [2/9] match_v2.py (region matcher $([ $REGION -eq 1 ] && echo on || echo off))"
  MATCH_ARGS=()
  [ "$REGION" -eq 1 ] || MATCH_ARGS+=(--no-region-match)
  $PYTHON "$V2/match_v2.py" "$WORK/kernel_debuf.mlir" -o "$WORK/matched.mlir" \
    --report "$WORK/match.json" "${MATCH_ARGS[@]}" 2>"$WORK/match.err" || {
      cat "$WORK/match.err" >&2; echo "ERROR: match_v2 failed" >&2; exit 1; }
  tail -n 1 "$WORK/match.err" | sed 's/^/         /'
else
  echo "  [2/9] matching disabled (residual control)"
  cp "$WORK/kernel_debuf.mlir" "$WORK/matched.mlir"
fi
fi

# ─── 3. lower kernel.launch -> polygeist_v2_<call> ──────────────────────
echo "  [3/9] lower_v2.py"
$PYTHON "$V2/lower_v2.py" "$WORK/matched.mlir" -o "$WORK/lowered.mlir" \
  --report "$WORK/lower.json" "${LOWER_ARGS[@]}" 2>"$WORK/lower.err" || {
    cat "$WORK/lower.err" >&2; echo "ERROR: lower_v2 failed" >&2; exit 1; }
sed 's/^/         /' "$WORK/lower.err"

# ─── 4. runtime freshness ───────────────────────────────────────────────
$PYTHON "$V2/gen_v2_runtime.py" --check 2>"$WORK/runtime_check.err" || {
  cat "$WORK/runtime_check.err" >&2
  echo "ERROR: runtime/polygeist_v2_rt* are stale; run gen_v2_runtime.py" >&2; exit 1; }

# ─── 5. remaining MLIR -> LLVM (polygeist_build.sh step 6, non-C-ABI) ───
echo "  [5/9] MLIR -> LLVM IR"
polygeist-opt --lower-polygeist-submap --canonicalize-polygeist \
  "$WORK/lowered.mlir" -o "$WORK/abi_canon.mlir" 2>"$WORK/mlir.err" || {
    cat "$WORK/mlir.err" >&2; exit 1; }
if grep -q 'polygeist\.submap' "$WORK/abi_canon.mlir"; then
  polygeist-opt --convert-linalg-to-loops --canonicalize --lower-polygeist-submap \
    --canonicalize "$WORK/abi_canon.mlir" -o "$WORK/abi_canon_loops.mlir" \
    2>>"$WORK/mlir.err" || { cat "$WORK/mlir.err" >&2; exit 1; }
  mv "$WORK/abi_canon_loops.mlir" "$WORK/abi_canon.mlir"
fi
sed -i 's|bufferization\.to_tensor \(%[^ ]*\) :|bufferization.to_tensor \1 restrict :|g' \
  "$WORK/abi_canon.mlir"
$MLIR_OPT --convert-math-to-llvm --empty-tensor-to-alloc-tensor --lower-affine \
  --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops --convert-scf-to-cf --expand-strided-metadata \
  --lower-affine --convert-arith-to-llvm --convert-index-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
  "$WORK/abi_canon.mlir" -o "$WORK/llvm.mlir" 2>>"$WORK/mlir.err" || {
    cat "$WORK/mlir.err" >&2; echo "ERROR: mlir-opt lowering failed" >&2; exit 1; }
$MLIR_TRANSLATE --mlir-to-llvmir "$WORK/llvm.mlir" -o "$WORK/kernel.ll"
sed -i "s/@${FUNCTION}\b/@${FUNCTION}_impl/g" "$WORK/kernel.ll"
N_CALL=$(grep -c 'call void @polygeist_v2_\|call double @polygeist_v2_' "$WORK/kernel.ll" || true)
echo "         $N_CALL polygeist_v2_* call site(s) in kernel.ll"
if [ "$TARGET" = jetson ]; then
  sed -i 's|target triple = "x86_64.*"|target triple = "aarch64-linux-gnu"|' "$WORK/kernel.ll"
  sed -i '/^target datalayout/d' "$WORK/kernel.ll"
fi

# ─── 6. C-ABI bridge ────────────────────────────────────────────────────
echo "  [6/9] wrapper"
$PYTHON "$SCRIPTS/gen_wrapper.py" "$INPUT" "$FUNCTION" > "$WORK/wrapper.c"

# ─── 7. harness ─────────────────────────────────────────────────────────
if [ -z "$HARNESS" ]; then
  echo "  [7/9] harness (generated: declares $FUNCTION)"
  $PYTHON "$SCRIPTS/generate_polybench_harness.py" "$INPUT" --function "$FUNCTION" \
    -o "$WORK/harness.c"
  HARNESS="$WORK/harness.c"
else
  echo "  [7/9] harness $HARNESS"
fi

# ─── 8. compile ─────────────────────────────────────────────────────────
echo "  [8/9] compile (target=$TARGET)"
HARNESS_EXTRA=()
POLYBENCH_FLAGS=()
for arg in "${PASS[@]}"; do
  [[ "$arg" == -Dstatic=* ]] && HARNESS_EXTRA+=(-fgnu89-inline) || POLYBENCH_FLAGS+=("$arg")
done
if [ "$TARGET" = host ]; then
  CC=$CLANG; CLANG_T=()
  RT_SRC=$RT/polygeist_v2_rt_cpu.c; RT_INC=()
  RT_LIBS=(${POLYGEIST_V2_CPU_BLAS_LIBS:--lopenblas} -lm -lpthread)
  NM=nm
else
  CC=$AARCH64_CC; CLANG_T=(--target=aarch64-linux-gnu --gcc-toolchain=/usr)
  RT_SRC=$RT/polygeist_v2_rt_cuda.c; RT_INC=(-I"$CUDA_CROSS/include")
  RT_LIBS=(-L"$CUDA_CROSS/lib" -L"$CUDA_CROSS/lib/stubs" -lcusolver -lcublas -lcudart
           -lm -lpthread -ldl -Wl,-rpath,/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu)
  NM=aarch64-linux-gnu-nm
fi
$CLANG "${CLANG_T[@]}" -O3 -c "$WORK/kernel.ll" -o "$WORK/kernel.o"
$CC -O2 "${PASS[@]}" -c "$WORK/wrapper.c" -o "$WORK/wrapper.o"
NOINLINE=(-fno-inline -fno-inline-functions -fsemantic-interposition)
"$CC" --version 2>/dev/null | head -1 | grep -qi clang && NOINLINE+=(-Wno-unknown-attributes)
"$CC" --version 2>/dev/null | head -1 | grep -qi clang || \
  NOINLINE+=(-fno-ipa-cp -fno-ipa-cp-clone -fno-ipa-sra)
$CC "${PASS[@]}" -O3 "${NOINLINE[@]}" "${HARNESS_EXTRA[@]}" -c "$HARNESS" -o "$WORK/harness.o"
if $NM "$WORK/harness.o" | awk '$2 ~ /^[TtWw]$/ {print $3}' | grep -qx "$FUNCTION"; then
  echo "ERROR: harness defines $FUNCTION (symbol replacement is forbidden)" >&2; exit 1
fi
$CC -O2 -ffunction-sections -fdata-sections -I"$RT" "${RT_INC[@]}" -c "$RT_SRC" -o "$WORK/rt_v2.o"
$CC -O2 -c "$RT/polygeist_mlir_runner_utils.c" -o "$WORK/mlir_runner_utils.o"
EXTRA_OBJS=()
if grep -q '#include\s*[<"]polybench.h[>"]' "$HARNESS"; then
  for arg in "${PASS[@]}"; do
    case "$arg" in -I*) d=${arg#-I}; [ -f "$d/polybench.c" ] && { PB_C="$d/polybench.c"; break; } ;; esac
  done
  if [ -n "${PB_C:-}" ]; then
    $CC -O2 "${POLYBENCH_FLAGS[@]}" -c "$PB_C" -o "$WORK/polybench.o"
    EXTRA_OBJS+=("$WORK/polybench.o")
  fi
fi

# ─── 9. link ────────────────────────────────────────────────────────────
echo "  [9/9] link -> $OUT"
$CC -O2 "$WORK/kernel.o" "$WORK/wrapper.o" "$WORK/harness.o" "$WORK/rt_v2.o" \
  "$WORK/mlir_runner_utils.o" "${EXTRA_OBJS[@]}" "${RT_LIBS[@]}" \
  -Wl,--gc-sections -o "$OUT"
$NM "$OUT" | grep -E ' [TU] (polygeist_v2_|[a-z]+_$|cublas|cusolver)' | \
  awk '{print "         " $0}' | head -20 || true
echo "═══ build complete: $OUT"
