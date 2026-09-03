#!/usr/bin/env python3
"""Generate torch-native benchmark specs at the EXACT shapes the resident
harness uses (driver cfg["dims"]) — the single source of truth so native and
raised-resident are measured at identical shape + dtype (f32). Output:
resident_shape_specs.json for bench_shaped.py. Run with /usr/bin/python3.10."""
import importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gs = _load("gs", "issues/aten_c_kernels/native_cuda_results/gen_shape_specs.py")
drv = _load("drv", "scripts/correctness/aten_pointwise_graph_silicon.py")


def main():
    specs = []
    for kernel in drv._matched_kernels():
        cfg = drv._cfg_for(kernel)
        if not cfg:
            continue
        base = gs.matched_base(kernel)
        cat = gs.CAT.get(base)
        if cat is None:
            continue  # no torch op to compare against
        dims = {k: int(v) for k, v in cfg["dims"].items()}
        # total element count = product of dims (matches resident data size for
        # pointwise; structured cats use the dims directly in bench_shaped).
        n = 1
        for v in dims.values():
            n *= max(1, v)
        shape = "_".join(f"{k}={v}" for k, v in cfg["dims"].items())
        specs.append({"kernel": kernel, "op": base, "cat": cat,
                      "dims": dims, "n": n, "shape": shape})
    out = Path(__file__).with_name("resident_shape_specs.json")
    out.write_text(json.dumps(specs, indent=0))
    print(f"wrote {out} with {len(specs)} native specs at resident shapes")


if __name__ == "__main__":
    main()
