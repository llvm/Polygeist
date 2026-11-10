#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/home/arjaiswal/Polygeist}
cuda=${CUDA_HOME:-/usr/local/cuda-12.6}
out=${OUT:-/tmp/mfem-isolated-mass2-raise}
cgeist=${CGEIST:-$repo/build/bin/cgeist}
opt=${POLYGEIST_OPT:-$repo/build/bin/polygeist-opt}
resource_dir=${CLANG_RESOURCE_DIR:-$repo/llvm-project/build/lib/clang/18}
source_file=$repo/issues/mfem_c_kernels/direct_native_cgeist/isolated_mass2_native.cu
matcher=$repo/scripts/correctness/kernel_match_rewrite.py

mkdir -p "$out"

"$cgeist" "$source_file" \
  --cuda-gpu-arch=sm_87 --cuda-path="$cuda" \
  --resource-dir="$resource_dir" --std=c++17 --function='*' \
  --cuda-lower --cpuify=distribute -DPOLYGEIST_TRANSLATION_ONLY \
  -I"$repo/issues/mfem_c_kernels/direct_native_cgeist/clang_cuda_stubs" \
  -I"$cuda/targets/sbsa-linux/include" -S -o "$out/frontend.mlir" \
  > "$out/frontend.stdout" 2> "$out/frontend.stderr"

"$opt" --canonicalize --cse --raise-scf-to-affine \
  "$out/frontend.mlir" -o "$out/affine.mlir" \
  > "$out/affine.stdout" 2> "$out/affine.stderr"

"$opt" --remove-iter-args --affine-parallelize \
  --raise-affine-to-linalg-pipeline "$out/affine.mlir" \
  -o "$out/raised.mlir" > "$out/raise.stdout" 2> "$out/raise.stderr"

"$opt" --linalg-debufferize=use-multi-root=true "$out/raised.mlir" \
  -o "$out/debufferized.mlir" \
  > "$out/debufferize.stdout" 2> "$out/debufferize.stderr"

/usr/bin/python3 "$matcher" "$out/debufferized.mlir" --dry-run \
  --show-candidates --show-semantic-only --show-structured-regions \
  > "$out/matcher.stdout" 2> "$out/matcher.stderr"

printf 'frontend_mulf=%s\n' "$(grep -c 'arith.mulf' "$out/frontend.mlir" || true)"
printf 'raised_linalg_generic=%s\n' "$(grep -c 'linalg.generic' "$out/raised.mlir" || true)"
printf 'raised_affine_loops=%s\n' "$(grep -Ec 'affine\.(for|parallel)' "$out/raised.mlir" || true)"
printf 'raised_scf_loops=%s\n' "$(grep -Ec 'scf\.(for|parallel|while)' "$out/raised.mlir" || true)"
cat "$out/matcher.stderr"
