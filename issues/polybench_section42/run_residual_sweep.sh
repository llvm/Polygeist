#!/usr/bin/env bash
set -u

repo=$(cd "$(dirname "$0")/../.." && pwd)
result_root="$repo/issues/polybench_section42"
work_root=/tmp/polybench-section42-residual

export PATH=/tmp/polygeist-section42-build/bin:$PATH
export MLIR_OPT=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-opt
export MLIR_TRANSLATE=/home/arjaiswal/Polygeist/llvm-project/build/bin/mlir-translate
export CLANG=/home/arjaiswal/Polygeist/llvm-project/build/bin/clang
export MLIR_LIBDIR=/home/arjaiswal/Polygeist/llvm-project/build/lib
export POLYBENCH_DATASET=LARGE
export POLYBENCH_COMPILE_OPT=-O3
export POLYBENCH_E2E_OUTPUT_ROOT="$work_root"

mkdir -p "$work_root"
while IFS=, read -r kernel category source _; do
  [[ "$kernel" == kernel || "$kernel" == gemm ]] && continue
  kernel_dir=$(dirname "$repo/$source")
  log_dir="$result_root/logs/$kernel"
  mkdir -p "$log_dir" "$result_root/ir/$kernel"
  command_file="$log_dir/large_residual.command"
  printf '%q ' timeout 600s bash "$repo/scripts/correctness/run_kernel_e2e.sh" \
    "$kernel_dir" "$kernel" --debuf > "$command_file"
  printf '\n' >> "$command_file"
  start=$(date -Is)
  timeout 600s /usr/bin/time -f 'wall_seconds=%e maxrss_kb=%M' \
    bash "$repo/scripts/correctness/run_kernel_e2e.sh" \
      "$kernel_dir" "$kernel" --debuf \
      > "$log_dir/large_residual.log" 2>&1
  rc=$?
  end=$(date -Is)
  printf '%s,%s,%s,%s\n' "$kernel" "$rc" "$start" "$end" | \
    tee -a "$result_root/logs/residual_sweep_summary.csv"
done < "$result_root/manifest.csv"
