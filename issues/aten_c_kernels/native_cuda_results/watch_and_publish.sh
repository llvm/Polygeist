#!/bin/bash
# Live-publish: while the resident sweep runs, merge new results into
# resident_silicon.csv and regenerate the ATen HTML every ~90s.
set -uo pipefail
cd /home/arjaiswal/Polygeist
source envsetup.sh 2>/dev/null
CSV=issues/aten_c_kernels/native_cuda_results/resident_silicon.csv
RESULTS=/tmp/res_full/results.txt
SWEEP_OUT="${1:-}"   # optional path to the sweep task output (for DONE detection)
prev=-1
for i in $(seq 1 100); do
  cur=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
  if [ "$cur" != "$prev" ]; then
    python3 - "$CSV" "$RESULTS" <<'PY'
import csv, re, sys
csvp, resp = sys.argv[1], sys.argv[2]
existing={}
try:
    existing={r["kernel"]:r for r in csv.DictReader(open(csvp))}
except OSError: pass
for line in open(resp):
    m=re.search(r"kernel=(\S+) warm_us=([0-9.]+) resident_us=([0-9.]+) errors=(\d+) max_error=\S+ shape=(\S+)", line)
    if m and int(m.group(4))==0:
        w,r=float(m.group(2)),float(m.group(3))
        existing[m.group(1)]={"kernel":m.group(1),"resident_us":f"{r:.3f}",
            "warm_us":f"{w:.3f}","device_speedup":f"{w/r:.3f}","shape":m.group(5),
            "errors":"0","hardware":"Jetson_AGX_Orin_sm87_CUDA12.6","date":"2026-08-30"}
rows=sorted(existing.values(),key=lambda x:x["kernel"])
with open(csvp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["kernel","resident_us","warm_us","device_speedup","shape","errors","hardware","date"])
    w.writeheader(); w.writerows(rows)
print(len(rows))
PY
    python3 scripts/correctness/build_ce_viewer.py --aten-only >/dev/null 2>&1
    echo "[watch] $(wc -l < "$CSV") kernels in CSV; HTML regenerated ($cur raw results)"
    prev=$cur
  fi
  if [ -n "$SWEEP_OUT" ] && grep -q "\[sweep\] DONE" "$SWEEP_OUT" 2>/dev/null; then
    echo "[watch] sweep DONE; final publish complete"; break
  fi
  sleep 90
done
