#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit_dir="$(cd "${script_dir}/.." && pwd)"
source_dir="${audit_dir}/worktrees/candl-parsed-reconstruction"
build_dir="${audit_dir}/build/candl-parsed-reconstruction"
log_file="${audit_dir}/logs/configure_parsed_idiom_reconstruction.log"

mkdir -p "${build_dir}" "${audit_dir}/logs"
exec > >(tee "${log_file}") 2>&1

printf 'timestamp=%s\n' "$(date -Is)"
printf 'classification=experimental reconstruction; never count as original artifact\n'
printf 'source_commit=%s\n' "$(git -C "${source_dir}" rev-parse HEAD)"
printf 'source_branch=%s\n' "$(git -C "${source_dir}" branch --show-current)"
printf 'clang_commit=%s\n' "$(git -C "${source_dir}/tools/clang" rev-parse HEAD)"
printf 'ghc=%s\n' "$(ghc --numeric-version)"
printf 'pypy=%s\n' "$(pypy --version 2>&1 | head -n 1)"

# X86-only reduces build cost. This does not change the host-side LLVM IR
# matcher pipeline, but it is recorded as a deviation from the demo README's
# default all-target configuration.
cmake -S "${source_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_TARGETS_TO_BUILD=X86
