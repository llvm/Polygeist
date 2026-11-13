#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${audit_dir}/../.." && pwd)"
output="${audit_dir}/experiment_versions.tsv"

record_git() {
  local name="$1"
  local path="$2"
  printf '%s\tcommit\t%s\n' "$name" "$(git -C "$path" rev-parse HEAD)"
  printf '%s\ttree\t%s\n' "$name" "$(git -C "$path" rev-parse HEAD^{tree})"
  printf '%s\torigin\t%s\n' "$name" "$(git -C "$path" remote get-url origin)"
  if test -n "$(git -C "$path" status --porcelain --untracked-files=no)"; then
    printf '%s\ttracked_worktree\tdirty\n' "$name"
  else
    printf '%s\ttracked_worktree\tclean\n' "$name"
  fi
}

{
  printf 'component\tfield\tvalue\n'
  record_git polygeist "${repo_root}"
  record_git snu_npb_mirror "${repo_root}/third_party/ginsbach-snu-npb"
  record_git snu_npb_second_mirror "${repo_root}/third_party/ginsbach-snu-npb-source"
  record_git parboil_mirror "${repo_root}/third_party/gpu-parboil"
  record_git author_idl_demo "${repo_root}/third_party/ginsbach-original/IDL-Demo"
  record_git related_candl_llvm "${repo_root}/third_party/ginsbach-original/llvm"
  record_git author_clang "${repo_root}/third_party/ginsbach-original/llvm/tools/clang"
  record_git parsed_idiom_reconstruction "${audit_dir}/worktrees/candl-parsed-reconstruction"
  printf 'host\tuname\t%s\n' "$(uname -a)"
  printf 'host\tcc\t%s\n' "$(cc --version | head -n 1)"
  printf 'host\tcxx\t%s\n' "$(c++ --version | head -n 1)"
  printf 'host\tcmake\t%s\n' "$(cmake --version | head -n 1)"
  printf 'host\tninja\t%s\n' "$(ninja --version)"
  printf 'host\tghc\t%s\n' "$(ghc --numeric-version)"
  printf 'host\tpypy\t%s\n' "$(pypy --version 2>&1 | head -n 1)"
} > "${output}"

printf 'wrote %s\n' "${output}"
