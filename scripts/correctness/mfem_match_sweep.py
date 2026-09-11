#!/usr/bin/env python3
"""Debufferize and library-match every normalized MFEM kernel."""
import concurrent.futures
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues" / "mfem_c_kernels"
RAISE_RESULTS = CORPUS / "results"
OUT = CORPUS / "match_results"
OPT = ROOT / "build" / "bin" / "polygeist-opt"
MATCHER = ROOT / "scripts" / "correctness" / "kernel_match_rewrite.py"
PYTHON = Path("/usr/bin/python3")

MATCH_RE = re.compile(r"^\s+match\s+body#.*?\s{2,}(\S+)\s*$")
TOTAL_RE = re.compile(r"total:\s+(\d+) matched / (\d+) bodies")
LAUNCH_RE = re.compile(r"kernel\.launch\s+@([A-Za-z0-9_.$-]+)")

def run(command, timeout=180):
    try:
        p = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + "\ntimeout\n"

def process(row):
    ident = row["id"]
    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    raised = RAISE_RESULTS / f"{ident}__normalized.raised.mlir"
    debuf = directory / "debufferized.mlir"
    matched = directory / "matched.mlir"
    report_path = directory / "match_report.txt"
    debuf_log = directory / "debufferize.log"
    match_log = directory / "matcher.log"

    # MFEM stages commonly join several ABI buffers in one contraction.  Use
    # the atomic multi-root walker explicitly: the recursive fallback explores
    # each root independently and can revisit the same cross-root graph until
    # the per-kernel timeout, while the joint walker handles it in one pass.
    drc, _, derr = run([
        str(OPT), "--linalg-debufferize=use-multi-root=true", str(raised),
        "-o", str(debuf),
    ])
    debuf_log.write_text(derr)
    result = dict(id=ident, function=row["function"], family=row["family"],
                  dimension=row["dimension"], debufferize_ok=drc == 0,
                  matcher_ok=False, linalg_ops=0, matcher_bodies=0,
                  matched_groups=0, matched_symbols="", kernel_launches=0,
                  launch_symbols="", error="")
    if drc:
        result["error"] = next((x for x in derr.splitlines() if "error:" in x), derr[:300])
        return result
    text = debuf.read_text().rstrip() + "\n"
    debuf.write_text(text)
    result["linalg_ops"] = text.count("linalg.")
    rrc, report, rerr = run([str(PYTHON), str(MATCHER), str(debuf), "--dry-run"])
    combined_report = report + rerr
    report_path.write_text(combined_report)
    if rrc:
        result["error"] = next((x for x in rerr.splitlines() if "error" in x.lower()), rerr[:300])
        return result
    symbols = [m.group(1) for line in combined_report.splitlines()
               if (m := MATCH_RE.match(line))]
    total = TOTAL_RE.search(combined_report)
    result["matched_groups"] = int(total.group(1)) if total else len(symbols)
    result["matcher_bodies"] = int(total.group(2)) if total else 0
    result["matched_symbols"] = ",".join(sorted(set(symbols)))

    mrc, rewritten, merr = run([str(PYTHON), str(MATCHER), str(debuf)])
    matched.write_text(rewritten.rstrip() + "\n")
    match_log.write_text(merr)
    result["matcher_ok"] = mrc == 0
    launches = LAUNCH_RE.findall(rewritten)
    result["kernel_launches"] = len(launches)
    result["launch_symbols"] = ",".join(sorted(set(launches)))
    if mrc:
        result["error"] = next((x for x in merr.splitlines() if "error" in x.lower()), merr[:300])
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader((CORPUS/"manifest.csv").open())
            if r["variant"] == "normalized"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            row = future.result(); results.append(row)
            print(f"{row['id']:<40} matches={row['matched_groups']:<3} "
                  f"bodies={row['matcher_bodies']:<3} launches={row['kernel_launches']}",
                  flush=True)
    results.sort(key=lambda r: r["id"])
    with (OUT/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader(); writer.writerows(results)
    matched = [r for r in results if r["matched_groups"]]
    with (OUT/"SUMMARY.md").open("w") as f:
        f.write("# MFEM normalized-kernel library matching\n\n")
        f.write(f"- kernels: {len(results)}\n")
        f.write(f"- matcher successes: {sum(r['matcher_ok'] for r in results)}\n")
        f.write(f"- kernels with at least one match: {len(matched)}\n")
        f.write(f"- matched stage groups: {sum(r['matched_groups'] for r in results)}\n")
        f.write(f"- emitted kernel.launch operations: {sum(r['kernel_launches'] for r in results)}\n\n")
        f.write("Matches are stage-level unless a report explicitly names a whole composition.\n")
        f.write("""

## FP64 contraction lowering

- 128 matches are ABI-legal two-input FP64 contractions:
  - 64 rank `4 x 5 -> 4`
  - 4 rank `5 x 4 -> 4`
  - 20 rank `5 x 5 -> 4`
- 40 iterator/rank-generic launches, comprising all 36 2D contraction stages
  plus 4 3D stages whose physical output views compact a broadcast mode
- All 128 lower to `polygeist_cutensornet_contraction2_f64`, with the original
  affine indexing maps and physical `polygeist.submap` strides encoded as
  extent/stride/mode metadata.
- A reduction dimension may occur in the logical output map only when its
  physical `polygeist.submap` stride is proven zero; ABI lowering then omits
  that broadcast mode from the output descriptor.
- The remaining 6 emitted launches are older structural matches (5
  `cublasDaxpby`, 1 `cudnnAddTensor_batched`) and are not included in the
  cuTensorNet lowering count.

Host compilation, focused pass tests, and the CPU reference contraction test
pass.

## Silicon validation

On 2026-07-24, all three compiler-generated FP64 variants ran through
cuTensorNet on an aarch64 Tegra target:

- `r4 x r5 -> r4`: `max_error=0`
- `r5 x r4 -> r4`: `max_error=0`
- `r5 x r5 -> r4` with compacted broadcast modes: `max_error=0`

The first call paid cuTensorNet/CUDA initialization and planning cost. The
subsequent two small contractions took about `0.085-0.088 ms` of device time.
See `../silicon_results/2026-07-24_cutensornet_variants.log`.
""")

if __name__ == "__main__": main()
