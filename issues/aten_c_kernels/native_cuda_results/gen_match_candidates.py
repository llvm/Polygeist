#!/usr/bin/env python3
"""Combine executable matches and candidate enumeration for every ATen kernel.

Record every symbol emitted by the current structured rewrite as well as all
ABI-lowerable and semantic-only alternatives, so the HTML can show the complete
library-match set rather than only the dry-run winner. Output: match_candidates.json
  { kernel: {"winner": sym|null, "candidates": [sym, ...]} }
Run with /usr/bin/python3 (needs egglog)."""
import concurrent.futures
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "issues/aten_c_kernels/results"
REWRITE = ROOT / "scripts/correctness/kernel_match_rewrite.py"
OUT = Path(__file__).with_name("match_candidates.json")
PYTHON = os.environ.get("ATEN_MATCH_PYTHON", "/usr/bin/python3")

CAND_RE = re.compile(
    r"(kernel_candidate|semantic_debug)\s+body#\[[^\]]*\]\s+(\S+)\s+kind=")
WIN_RE = re.compile(
    r"^\s*match\s+body#\[[^\]]*\]\s+([A-Za-z0-9_]+)"
)
EMITTED_RE = re.compile(r"kernel\.launch @([A-Za-z0-9_]+)")


def one(debuf: Path):
    kernel = debuf.parent.name
    matched = debuf.with_name("matched.mlir")
    try:
        r = subprocess.run(
            [PYTHON, str(REWRITE), "--dry-run", "--show-candidates",
             "--show-semantic-only", "--enable-structured-rewrite",
             str(debuf)],
            capture_output=True, text=True, timeout=60)
        out = r.stderr + "\n" + r.stdout  # the match report prints to stderr
    except Exception as error:
        return kernel, None, str(error)
    if r.returncode:
        return kernel, None, f"matcher exited {r.returncode}: {out[-500:]}"
    # Structured executable rewriting may select a specialized whole-region
    # implementation that differs from the generic dry-run winner.  The
    # rematched artifact is therefore authoritative for emitted symbols; the
    # report contributes the remaining alternatives.
    emitted = list(dict.fromkeys(
        EMITTED_RE.findall(matched.read_text()) if matched.exists() else []
    ))
    cands = [{"name": name, "abi": True} for name in emitted]
    seen = set(emitted)
    winner = emitted[0] if emitted else None
    for line in out.splitlines():
        m = CAND_RE.search(line)
        if m and m.group(2) not in seen:
            seen.add(m.group(2))
            cands.append({"name": m.group(2),
                          "abi": m.group(1) == "kernel_candidate"})
        w = WIN_RE.match(line)
        if w:
            report_winner = w.group(1)
            if winner is None:
                winner = report_winner
            if report_winner in seen:
                for candidate in cands:
                    if candidate["name"] == report_winner:
                        candidate["abi"] = True
                        break
            else:
                seen.add(report_winner)
                cands.append({"name": report_winner, "abi": True})
    return kernel, {"winner": winner, "candidates": cands}, None


def main():
    debufs = sorted(RESULTS.glob("*/debuf.mlir"))
    result = {}
    failures = []
    workers = int(os.environ.get("ATEN_MATCH_CANDIDATE_WORKERS", "4"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for kernel, data, error in pool.map(one, debufs):
            if data is not None:
                result[kernel] = data
            else:
                failures.append(f"{kernel}: {error}")
    if failures:
        raise SystemExit(
            "candidate enumeration failed; existing JSON was preserved:\n"
            + "\n".join(failures)
        )
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=0) + "\n")
    temporary.replace(OUT)
    multi = sum(1 for v in result.values() if len(v["candidates"]) > 1)
    multi_abi = sum(
        1 for v in result.values()
        if sum(1 for c in v["candidates"] if c["abi"]) > 1)
    print(f"wrote {OUT} for {len(result)} kernels; "
          f"{multi} have >1 candidate, {multi_abi} have >1 abi-lowerable")


if __name__ == "__main__":
    main()
