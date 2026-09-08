#!/usr/bin/env bash
set -uo pipefail

campaign_root=${1:?usage: run_mfem_native_cuda_campaign.sh CAMPAIGN_ROOT}
bin_dir="$campaign_root/bin"
raw_dir="$campaign_root/raw"
metadata_dir="$campaign_root/metadata"
mkdir -p "$raw_dir" "$metadata_dir"

ids=(ag2 ag3 av2 conv2 conv3 curl2 curl3 diff2 diff3 div2 div3 ig2 ig3 iv2 iv3 mass2 mass3)
total=$((5 * ${#ids[@]}))
completed=0
export LD_LIBRARY_PATH="$bin_dir:/home/nvidia/cuda-12.6/lib64:/home/nvidia/cuda-12.6/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
  printf 'start_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'hostname=%s\n' "$(hostname)"
  uname -a
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
} >"$metadata_dir/board-start.txt"
sha256sum "$bin_dir"/* >"$metadata_dir/binary.sha256"
: >"$campaign_root/progress.log"

for process in 0 1 2 3 4; do
  for id in "${ids[@]}"; do
    final="$raw_dir/process${process}_${id}_mfem_native_cuda.log"
    temporary="$final.tmp"
    if [[ -f $final ]] && grep -q '^RUN_STATUS=PASS$' "$final"; then
      completed=$((completed + 1))
      continue
    fi
    printf 'START completed=%d/%d process=%d executable=%s\n' \
      "$completed" "$total" "$process" "$id" | tee -a "$campaign_root/progress.log"
    {
      printf 'process=%d\nexecutable=%s_mfem_native_cuda\nstart_utc=%s\n' \
        "$process" "$id" "$(date -u +%FT%TZ)"
      timeout --signal=TERM --kill-after=10s 15m \
        "$bin_dir/${id}_mfem_native_cuda"
      status=$?
      printf 'exit_code=%d\nend_utc=%s\n' "$status" "$(date -u +%FT%TZ)"
      samples=$(grep -c '^implementation=' "$temporary" 2>/dev/null || true)
      if [[ $status -eq 0 && $samples -eq 20 ]]; then
        printf 'RUN_STATUS=PASS\n'
      else
        printf 'RUN_STATUS=FAIL\n'
      fi
    } >"$temporary" 2>&1
    samples=$(grep -c '^implementation=' "$temporary" 2>/dev/null || true)
    if grep -q '^RUN_STATUS=PASS$' "$temporary"; then
      mv "$temporary" "$final"
      completed=$((completed + 1))
      printf 'DONE completed=%d/%d process=%d executable=%s samples=%d\n' \
        "$completed" "$total" "$process" "$id" "$samples" | tee -a "$campaign_root/progress.log"
    else
      mv "$temporary" "$final.failed"
      printf 'FAIL completed=%d/%d process=%d executable=%s samples=%d\n' \
        "$completed" "$total" "$process" "$id" "$samples" | tee -a "$campaign_root/progress.log"
    fi
  done
done

{
  printf 'end_utc=%s\n' "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
} >"$metadata_dir/board-end.txt"
printf 'CAMPAIGN_COMPLETE completed=%d/%d\n' "$completed" "$total" | tee -a "$campaign_root/progress.log"
[[ $completed -eq $total ]]
