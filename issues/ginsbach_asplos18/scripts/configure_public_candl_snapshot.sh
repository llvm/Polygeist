#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${audit_dir}/../.." && pwd)"
llvm_src="${repo_root}/third_party/ginsbach-original/llvm"
clang_src="${llvm_src}/tools/clang"
build_dir="${audit_dir}/build/candl-public-original"
log_dir="${audit_dir}/logs"
log_file="${log_dir}/configure_public_candl_snapshot.log"

mkdir -p "${build_dir}" "${log_dir}"
exec > >(tee "${log_file}") 2>&1

printf 'timestamp=%s\n' "$(date -Is)"
printf 'scope=related public cc18ginsbach snapshot; not the missing ASPLOS artifact\n'
printf 'host=%s\n' "$(uname -a)"
printf 'llvm_commit=%s\n' "$(git -C "${llvm_src}" rev-parse HEAD)"
printf 'clang_commit=%s\n' "$(git -C "${clang_src}" rev-parse HEAD)"
printf 'cmake=%s\n' "$(cmake --version | head -n 1)"
printf 'ninja=%s\n' "$(ninja --version)"
printf 'cc=%s\n' "$(cc --version | head -n 1)"
printf 'cxx=%s\n' "$(c++ --version | head -n 1)"
printf 'ghc=%s\n' "$(command -v ghc || printf unavailable)"
printf 'pypy=%s\n' "$(command -v pypy || printf unavailable)"

# This is the exact CMake configuration documented by the author's IDL-Demo
# README, adjusted only for explicit source/build paths.
cmake -S "${llvm_src}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DLLVM_PARALLEL_LINK_JOBS=1
