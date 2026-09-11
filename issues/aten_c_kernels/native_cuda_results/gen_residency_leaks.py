#!/usr/bin/env python3
"""Step-0 residency-leak checker. For every ATen kernel's abi.mlir, count the
buffer allocs/copies the lowered code carries — especially between shim calls,
where they break device residency in a chain. Classifies each memref.copy as
ELIDABLE (copy-to-output, or single-use source -> store-forwardable) vs GENUINE
(a real layout/repack). Output: residency_leaks.json
  { kernel: {shim_calls, copies, inter_call_copies, elidable, genuine,
             allocs, inter_call_allocs} }
Run with plain python3."""
import json, re, subprocess, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "issues/aten_c_kernels/results"
OUT = Path(__file__).with_name("residency_leaks.json")
MLIR_OPT = os.environ.get("MLIR_OPT",
                          str(ROOT / "llvm-project/build/bin/mlir-opt"))

_TO_TENSOR = re.compile(r"bufferization\.to_tensor (%[^ ]*) :")


def bufferize(abi: Path):
    """Run the same one-shot-bufferize the jetson build uses; return the
    bufferized MLIR text, or None if bufferization fails (some kernels don't
    bufferize standalone outside the full pipeline)."""
    text = abi.read_text()
    text = _TO_TENSOR.sub(r"bufferization.to_tensor \1 restrict :", text)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(text)
        src = f.name
    try:
        r = subprocess.run(
            [MLIR_OPT, "--empty-tensor-to-alloc-tensor",
             "--one-shot-bufferize=bufferize-function-boundaries",
             "--canonicalize", "--promote-buffers-to-stack",
             src],  # matches build_jetson.sh [2/6]
            capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None
    finally:
        os.unlink(src)

ARG_RE = re.compile(r"(%arg\d+)\s*:")
CALL_RE = re.compile(r"\bcall @polygeist")
ALLOC_RE = re.compile(r"=\s*memref\.alloc\b")          # heap alloc (not alloca)
COPY_RE = re.compile(r"memref\.copy\s+(%\S+),\s*(%\S+)\s*:")
NAME_RE = re.compile(r"%[A-Za-z0-9_]+")


def analyze(abi: Path):
    # Prefer the bufferized IR (allocs/copies only materialize there); fall
    # back to the pre-bufferization abi.mlir when standalone bufferize fails.
    buffered = bufferize(abi)
    text = buffered if buffered is not None else abi.read_text()
    stage = "bufferized" if buffered is not None else "abi"
    # kernel body = the one non-private func.func with a body
    m = re.search(r"func\.func @[\w.]+\([^)]*\)[^{]*\{", text)
    if not m:
        return None
    args = set(ARG_RE.findall(text[m.start():m.end()]))
    body = text[m.end():]
    lines = body.splitlines()
    # occurrence count of every SSA name in the body (def + uses)
    uses = {}
    for tok in NAME_RE.findall(body):
        uses[tok] = uses.get(tok, 0) + 1

    call_idx = [i for i, ln in enumerate(lines) if CALL_RE.search(ln)]
    first, last = (call_idx[0], call_idx[-1]) if call_idx else (-1, -1)

    copies = allocs = inter_copies = inter_allocs = elidable = genuine = 0
    for i, ln in enumerate(lines):
        inter = first != -1 and first < i < last
        if ALLOC_RE.search(ln):
            allocs += 1
            inter_allocs += 1 if inter else 0
        cm = COPY_RE.search(ln)
        if cm:
            copies += 1
            inter_copies += 1 if inter else 0
            src, dst = cm.group(1), cm.group(2)
            # copy-to-output: dst is a function arg; or single-use source
            # (src appears only at its def and this copy -> store-forwardable)
            is_out = dst in args
            single_use = uses.get(src, 0) <= 2
            if is_out or single_use:
                elidable += 1
            else:
                genuine += 1
    return {
        "stage": stage,
        "shim_calls": len(call_idx),
        "copies": copies, "inter_call_copies": inter_copies,
        "elidable": elidable, "genuine": genuine,
        "allocs": allocs, "inter_call_allocs": inter_allocs,
    }


def main():
    import concurrent.futures
    abis = sorted(RESULTS.glob("*/abi.mlir"))
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for abi, data in zip(abis, pool.map(analyze, abis)):
            if data:
                result[abi.parent.name] = data
    OUT.write_text(json.dumps(result, indent=0))
    buffed = sum(1 for v in result.values() if v.get("stage") == "bufferized")
    print(f"bufferized {buffed}/{len(result)} (rest fell back to abi.mlir)")
    tot_copies = sum(v["copies"] for v in result.values())
    tot_elid = sum(v["elidable"] for v in result.values())
    tot_allocs = sum(v["allocs"] for v in result.values())
    multi = {k: v for k, v in result.items() if v["shim_calls"] > 1}
    multi_leaks = sum(v["inter_call_copies"] + v["inter_call_allocs"]
                      for v in multi.values())
    print(f"kernels: {len(result)}  ({len(multi)} multi-call)")
    print(f"total memref.copy: {tot_copies}  ({tot_elid} elidable, "
          f"{tot_copies - tot_elid} genuine)")
    print(f"total memref.alloc: {tot_allocs}")
    print(f"inter-call leaks in multi-call kernels: {multi_leaks}")
    # worst offenders
    worst = sorted(result.items(),
                   key=lambda kv: kv[1]["copies"] + kv[1]["allocs"],
                   reverse=True)[:12]
    print("\nworst (copies+allocs):")
    for k, v in worst:
        print(f"  {k:38s} calls={v['shim_calls']} copies={v['copies']}"
              f"(elid {v['elidable']}) allocs={v['allocs']}"
              f" inter={v['inter_call_copies']+v['inter_call_allocs']}")


if __name__ == "__main__":
    main()
