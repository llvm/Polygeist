#!/usr/bin/env python3
"""Generate per-kernel torch benchmark specs so each native measurement uses
the SAME size as its raised variant (the 'large problem' column).

Output: shape_specs.json — list of {kernel, op, cat, dims} for every ATen
kernel that renders a raised/native ratio. Consumed by bench_shaped.py on the
Jetson. No torch needed here."""
import csv, importlib.util, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BCV = ROOT / "scripts/correctness/build_ce_viewer.py"
spec = importlib.util.spec_from_file_location("bcv", BCV)
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except SystemExit:
    pass

LP = ROOT / "issues/aten_c_kernels/silicon_results/large_problem_comparison.csv"

# base torch op -> category
UNARY = {"abs","abs_complex","acos","acosh","asin","asinh","atan","atanh","ceil",
 "conj","cos","cosh","exp","exp2","floor","frac","log","neg","reciprocal","relu",
 "rsqrt","sigmoid","silu","sin","sinc","sinh","sqrt","square","tan","tanh","mish",
 "gelu","hardswish","leaky_relu","elu","erf","erfc","softplus"}
BINARY = {"add","mul","div","pow","hypot","logaddexp","logaddexp2"}
CAT = {op: "unary" for op in UNARY}
CAT.update({op: "binary" for op in BINARY})
CAT.update({
 "mse_loss":"loss","smooth_l1_loss":"loss","binary_cross_entropy":"loss_bce",
 "sum":"reduce","count_nonzero":"reduce","all":"reduce",
 "argmax":"reduce_arg","argmin":"reduce_arg","norm":"norm",
 "cumsum":"cum","cumprod":"cum","dot":"dot","sort":"sort","topk":"topk",
 "mm":"mm","addmm":"addmm","mv":"gemv","bmm":"bmm",
 "conv2d":"conv2d","conv3d":"conv3d","conv_transpose2d":"convT2d",
 "adaptive_avg_pool2d":"adaptavg2d","adaptive_max_pool2d":"adaptmax2d",
 "adaptive_avg_pool3d":"adaptavg3d","avg_pool2d":"avgpool2d","avg_pool3d":"avgpool3d",
 "max_pool2d":"maxpool2d","max_pool3d":"maxpool3d",
 "batch_norm":"batchnorm","layer_norm":"layernorm","rms_norm":"rmsnorm",
 "softmax":"softmax","cat":"cat",
})


def matched_base(kernel):
    if kernel in m._NATIVE_CUDA_US:
        return m._native_base(kernel)
    b = m._native_base(kernel)
    if b in m._NATIVE_CUDA_BASE:
        return b
    for mb in m._NATIVE_CUDA_BASE:
        if len(mb) < 3:
            continue
        bt, mbt = b.split("_"), mb.split("_")
        if (mb in bt or b in mbt or b.startswith(mb+"_") or b.endswith("_"+mb)
                or mb.startswith(b+"_") or mb.endswith("_"+b)):
            return mb
    return None


def parse(shape):
    """Extract labeled ints. Handles I=6x7 lists, N=K=<n>, bare 'N8388608'."""
    d = {}
    # chained equals: N=K=16777216
    for a, b, n in re.findall(r"([A-Za-z]+)=([A-Za-z]+)=(\d+)", shape):
        d[a] = int(n); d[b] = int(n)
    # I=6x7 / O=3x3x3 lists
    for key, lst in re.findall(r"([A-Za-z]+)=(\d+(?:x\d+)+)", shape):
        d[key] = [int(x) for x in lst.split("x")]
    # labeled scalars  LABEL=123 or LABEL123
    for key, n in re.findall(r"([A-Za-z]+)=?(\d+)\b", shape):
        if key not in d:
            d[key] = int(n)
    return d


def elems(d):
    if "N" in d and isinstance(d["N"], int):
        return d["N"]
    if "K" in d and isinstance(d["K"], int) and len(d) <= 2:
        return d["K"]
    prod = 1
    for k in ("B", "C", "H", "W", "D"):
        if isinstance(d.get(k), int):
            prod *= d[k]
    return prod if prod > 1 else 4194304


def build_specs():
    rows = {r["kernel"]: r for r in csv.DictReader(open(LP))}
    specs = []
    skipped = []
    for k, r in rows.items():
        if r.get("raised_us", "").strip() in ("", "—"):
            continue
        nat, _ = m.native_cuda_for(k)
        if not nat:
            continue
        base = matched_base(k)
        cat = CAT.get(base)
        shape = r.get("problem", "").strip()
        if cat is None or shape in ("", "—"):
            skipped.append((k, base, shape))
            continue
        d = parse(shape)
        specs.append({"kernel": k, "op": base, "cat": cat,
                      "dims": d, "n": elems(d), "shape": shape})
    return specs, skipped


if __name__ == "__main__":
    specs, skipped = build_specs()
    out = Path(__file__).with_name("shape_specs.json")
    out.write_text(json.dumps(specs, indent=0))
    print(f"wrote {out} with {len(specs)} specs; skipped {len(skipped)}")
    from collections import Counter
    c = Counter(s["cat"] for s in specs)
    for cat, n in c.most_common():
        print(f"  {cat:12s} {n}")
    if "--show-skip" in sys.argv:
        for k, b, s in skipped:
            print(f"  SKIP {k:34s} base={b} shape={s!r}")
