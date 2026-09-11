#!/bin/bash
# Full device-resident silicon sweep: build every buildable ATen CASE, run each
# on the Orin, collect resident_us + warm_us. Reduction kernels that hit the
# cuTensorNet blocker just fail to build and are skipped.
set -uo pipefail
cd /home/arjaiswal/Polygeist
source envsetup.sh 2>/dev/null
OUT=/tmp/res_full
rm -rf "$OUT"; mkdir -p "$OUT"
RESULTS="$OUT/results.txt"; : > "$RESULTS"

echo "[sweep] building ALL matched kernels (CASES + auto_spec)..."
/usr/bin/python3.10 scripts/correctness/aten_pointwise_graph_silicon.py \
  --all-matched --output "$OUT" --jobs 8 > "$OUT/build.log" 2>&1
built=$(ls -d "$OUT"/aten_*/ 2>/dev/null | while read d; do k=$(basename "$d"); [ -f "$d/$k" ] && echo "$k"; done)
n=$(echo "$built" | grep -c . || echo 0)
echo "[sweep] $n binaries built; running on Orin..."

DONE_CSV=/home/arjaiswal/Polygeist/issues/aten_c_kernels/native_cuda_results/resident_silicon.csv
i=0
for k in $built; do
  i=$((i+1))
  # skip kernels already measured (incremental re-runs)
  if [ "${SKIP_DONE:-1}" = "1" ] && grep -q "^$k," "$DONE_CSV" 2>/dev/null; then
    echo "[sweep] ($i/$n) $k: already measured, skip"; continue
  fi
  ST_TRACKER_JETSON_HOST=192.168.57.1 POLYGEIST_SILICON_PROFILE=pva-general \
    POLYGEIST_JETSON_LD_LIBRARY_PATH="/home/nvidia/cuda-12.6/lib64:/home/nvidia/cuda-12.6/targets/sbsa-linux/lib:/home/nvidia/cuda-12.6/targets/aarch64-linux/lib:/home/nvidia/polygeist_cuda_libs:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:/home/nvidia/venv/lib/python3.12/site-packages/nvidia/cudnn/lib" \
    POLYGEIST_JETSON_RUNS=3 timeout 200 scripts/correctness/run_jetson.sh \
    --exe "$OUT/$k/$k" "res_$k" > "$OUT/$k.run.log" 2>&1
  line=$(grep "RESULT" "$OUT/$k.run.log" | tail -1)
  [ -n "$line" ] && echo "$line" >> "$RESULTS"
  echo "[sweep] ($i/$n) $k: ${line:-NO_RESULT}"
done
echo "[sweep] DONE: $(wc -l < "$RESULTS") kernels produced a resident number"
