#!/bin/bash
set -uo pipefail

campaign_root=${1:?usage: run_mfem_application_campaign.sh CAMPAIGN_ROOT [jetson|jetson-cpu]}
target=${2:-jetson}
case "$target" in jetson|jetson-cpu) ;; *) echo "invalid target: $target" >&2; exit 2;; esac
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
suffix=
if [[ "$target" == jetson-cpu ]]; then suffix=-raised-cpu; fi
bin_dir="$campaign_root/bin$suffix"
raw_dir="$campaign_root/raw$suffix"
metadata_dir="$campaign_root/metadata"
mkdir -p "$raw_dir" "$metadata_dir"

ids=(
  mtop_iso_elasticity_dfem_2d dfem_minimal_surface_2d ex35p_h1_3d
  ex35p_hcurl_3d ex35p_hdiv_3d ex9p_mass_convection_2d grad_div_3d
  abs_l1_mass_3d abs_l1_diffusion_3d abs_l1_curlcurl_3d
  navier_tgv_pa_operators_3d
)
extra_libs=${POLYGEIST_JETSON_EXTRA_LIBS:-'/tmp/cutensornet_aarch64/lib/libcutensornet.so.2 /tmp/cutensornet_aarch64/lib/libcutensor.so.2'}
if [[ "$target" == jetson-cpu ]]; then extra_libs=; fi
: > "$campaign_root/run-progress$suffix.log"

failures=0
for id in "${ids[@]}"; do
  if [[ ! -x "$bin_dir/$id" ]]; then
    printf 'SKIP id=%s reason=missing-binary\n' "$id" | \
      tee -a "$campaign_root/run-progress$suffix.log"
    failures=$((failures + 1))
    continue
  fi
  printf 'START id=%s utc=%s\n' "$id" "$(date -u +%FT%TZ)" | \
    tee -a "$campaign_root/run-progress$suffix.log"
  if POLYGEIST_SILICON_PROFILE=pva-general \
      POLYGEIST_JETSON_RUNS=5 \
      POLYGEIST_LOG_DIR="$raw_dir" \
      POLYGEIST_JETSON_EXTRA_LIBS="$extra_libs" \
      "$repo_root/scripts/correctness/run_jetson.sh" \
      --exe "$bin_dir/$id" "mfem_derived${suffix}_$id" \
      > "$raw_dir/$id.deploy.log" 2>&1; then
    printf 'PASS id=%s utc=%s\n' "$id" "$(date -u +%FT%TZ)" | \
      tee -a "$campaign_root/run-progress$suffix.log"
  else
    failures=$((failures + 1))
    printf 'FAIL id=%s log=%s utc=%s\n' "$id" \
      "$raw_dir/$id.deploy.log" "$(date -u +%FT%TZ)" | \
      tee -a "$campaign_root/run-progress$suffix.log"
  fi
done

printf 'RUN_COMPLETE passed=%d failed=%d\n' \
  "$(( ${#ids[@]} - failures ))" "$failures" | \
  tee -a "$campaign_root/run-progress$suffix.log"
[[ $failures -eq 0 ]]
