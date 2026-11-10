#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${audit_dir}/build/candl-parsed-reconstruction"
log_file="${audit_dir}/logs/build_parsed_idiom_reconstruction.log"

exec > >(tee "${log_file}") 2>&1
printf 'timestamp=%s\n' "$(date -Is)"
printf 'classification=experimental reconstruction; never count as original artifact\n'
ninja -C "${build_dir}" clang -v
