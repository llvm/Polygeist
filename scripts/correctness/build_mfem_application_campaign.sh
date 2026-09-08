#!/bin/bash
set -uo pipefail

campaign_root=${1:?usage: build_mfem_application_campaign.sh CAMPAIGN_ROOT [NE] [jetson|jetson-cpu]}
ne=${2:-1024}
target=${3:-jetson}
case "$target" in jetson|jetson-cpu) ;; *) echo "invalid target: $target" >&2; exit 2;; esac
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
app_dir="$repo_root/issues/mfem_c_kernels/application_extractions"
suffix=
extra_define=()
if [[ "$target" == jetson-cpu ]]; then
  suffix=-raised-cpu
  extra_define=(-DMFEM_RAISED_CPU)
fi
bin_dir="$campaign_root/bin$suffix"
log_dir="$campaign_root/build-logs$suffix"
work_dir="$campaign_root/build-work$suffix"
metadata_dir="$campaign_root/metadata"
mkdir -p "$bin_dir" "$log_dir" "$work_dir" "$metadata_dir"

cutensornet_root=${POLYGEIST_CUTENSORNET_ROOT:-/tmp/cutensornet_aarch64}
export POLYGEIST_CUTENSORNET_ROOT="$cutensornet_root"
export POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME=1
# The Mass3D network composition is not correctness-approved. Use the same
# pairwise cuTensorNet lowering as the validated standalone-kernel campaign.
export POLYGEIST_COMPOSE_CUTENSORNET_NETWORKS=0

entries=(
  'mtop_iso_elasticity_dfem_2d|mtop_iso_elasticity_dfem_2d.c|mfem_app_mtop_iso_elasticity_dfem_2d|MFEM_APP_MTOP'
  'dfem_minimal_surface_2d|dfem_minimal_surface_2d.c|mfem_app_dfem_minimal_surface_2d|MFEM_APP_MINIMAL_SURFACE'
  'ex35p_h1_3d|ex35p_pa_operators.c|mfem_app_ex35p_h1_3d|MFEM_APP_EX35P_H1'
  'ex35p_hcurl_3d|ex35p_pa_operators.c|mfem_app_ex35p_hcurl_3d|MFEM_APP_EX35P_HCURL'
  'ex35p_hdiv_3d|ex35p_pa_operators.c|mfem_app_ex35p_hdiv_3d|MFEM_APP_EX35P_HDIV'
  'ex9p_mass_convection_2d|ex9p_mass_convection_2d.c|mfem_app_ex9p_mass_convection_2d|MFEM_APP_EX9P'
  'grad_div_3d|grad_div_3d.c|mfem_app_grad_div_3d|MFEM_APP_GRAD_DIV'
  'abs_l1_mass_3d|abs_l1_jacobi_operators.c|mfem_app_abs_l1_mass_3d|MFEM_APP_ABS_MASS'
  'abs_l1_diffusion_3d|abs_l1_jacobi_operators.c|mfem_app_abs_l1_diffusion_3d|MFEM_APP_ABS_DIFFUSION'
  'abs_l1_curlcurl_3d|abs_l1_jacobi_operators.c|mfem_app_abs_l1_curlcurl_3d|MFEM_APP_ABS_CURLCURL'
  'navier_tgv_pa_operators_3d|navier_tgv_pressure_diffusion_3d.c|mfem_app_navier_tgv_pa_operators_3d|MFEM_APP_NAVIER'
)

{
  printf 'generated_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'git_head=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
  printf 'git_status_porcelain_sha256='
  git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}'
  printf 'ne=%s\ntarget=%s\n' "$ne" "$target"
  printf 'cutensornet_root=%s\n' "$cutensornet_root"
  sha256sum "$repo_root/build/bin/cgeist" "$repo_root/build/bin/polygeist-opt" \
    "$repo_root/scripts/correctness/kernel_match_rewrite.py" \
    "$repo_root/runtime/polygeist_cublas_rt_cuda.c"
} > "$metadata_dir/build$suffix.txt"

: > "$campaign_root/build-progress$suffix.log"
failures=0
for entry in "${entries[@]}"; do
  IFS='|' read -r id source function define <<< "$entry"
  printf 'START id=%s utc=%s\n' "$id" "$(date -u +%FT%TZ)" | \
    tee -a "$campaign_root/build-progress$suffix.log"
  export POLYGEIST_BUILD_WORK_DIR="$work_dir/$id"
  if "$repo_root/scripts/correctness/polygeist_build.sh" \
      --target="$target" --function="$function" \
      --harness="$app_dir/mfem_application_reference_harness.c" \
      "$app_dir/$source" "-D$define" "-DMFEM_BENCH_NE=$ne" \
      -DMFEM_PUBLICATION_PROTOCOL -DMFEM_PUBLICATION_WARMUPS=5 \
      -DMFEM_PUBLICATION_SAMPLES=20 "${extra_define[@]}" -o "$bin_dir/$id" \
      > "$log_dir/$id.log" 2>&1; then
    launches=$(sed -n 's/.*matched \([0-9][0-9]*\) kernel.launch.*/\1/p' \
      "$log_dir/$id.log" | tail -1)
    calls=$(sed -n 's/.*emitted \([0-9][0-9]*\) func.call.*/\1/p' \
      "$log_dir/$id.log" | tail -1)
    printf 'PASS id=%s launches=%s calls=%s utc=%s\n' \
      "$id" "${launches:-unknown}" "${calls:-unknown}" "$(date -u +%FT%TZ)" | \
      tee -a "$campaign_root/build-progress$suffix.log"
  else
    failures=$((failures + 1))
    printf 'FAIL id=%s log=%s utc=%s\n' "$id" "$log_dir/$id.log" \
      "$(date -u +%FT%TZ)" | tee -a "$campaign_root/build-progress$suffix.log"
  fi
done

sha256sum "$bin_dir"/* > "$metadata_dir/binaries$suffix.sha256" 2>/dev/null || true
printf 'BUILD_COMPLETE passed=%d failed=%d\n' \
  "$(( ${#entries[@]} - failures ))" "$failures" | \
  tee -a "$campaign_root/build-progress$suffix.log"
[[ $failures -eq 0 ]]
