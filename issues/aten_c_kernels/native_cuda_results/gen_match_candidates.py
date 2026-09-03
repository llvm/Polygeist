#!/usr/bin/env python3
"""Run the matcher's candidate enumeration for every ATen kernel and record
ALL abi-lowerable candidates per body (not just the greedy winner), so the HTML
can show what else could have matched. Output: match_candidates.json
  { kernel: {"winner": sym|null, "candidates": [sym, ...]} }
Run with /usr/bin/python3.10 (needs egglog)."""
import json, re, subprocess, concurrent.futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "issues/aten_c_kernels/results"
REWRITE = ROOT / "scripts/correctness/kernel_match_rewrite.py"
OUT = Path(__file__).with_name("match_candidates.json")

CAND_RE = re.compile(
    r"(kernel_candidate|semantic_debug)\s+body#\[[^\]]*\]\s+(\S+)\s+kind=")
WIN_RE = re.compile(r"^\s*match\s+body#\[[^\]]*\]\s+(\S+)\s*$")


def one(debuf: Path):
    kernel = debuf.parent.name
    try:
        r = subprocess.run(
            ["/usr/bin/python3.10", str(REWRITE), "--dry-run",
             "--show-candidates", "--show-semantic-only", str(debuf)],
            capture_output=True, text=True, timeout=60)
        out = r.stderr + "\n" + r.stdout  # the match report prints to stderr
    except Exception:
        return kernel, None
    cands, seen, winner = [], set(), None
    for line in out.splitlines():
        m = CAND_RE.search(line)
        if m and m.group(2) not in seen:
            seen.add(m.group(2))
            cands.append({"name": m.group(2),
                          "abi": m.group(1) == "kernel_candidate"})
        w = WIN_RE.match(line)
        if w:
            winner = w.group(1)
    return kernel, {"winner": winner, "candidates": cands}


def main():
    debufs = sorted(RESULTS.glob("*/debuf.mlir"))
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for kernel, data in pool.map(one, debufs):
            if data is not None:
                result[kernel] = data
    OUT.write_text(json.dumps(result, indent=0))
    multi = sum(1 for v in result.values() if len(v["candidates"]) > 1)
    multi_abi = sum(
        1 for v in result.values()
        if sum(1 for c in v["candidates"] if c["abi"]) > 1)
    print(f"wrote {OUT} for {len(result)} kernels; "
          f"{multi} have >1 candidate, {multi_abi} have >1 abi-lowerable")


if __name__ == "__main__":
    main()
