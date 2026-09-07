#!/usr/bin/env python3
"""Create the reproducible ATen benchmark coverage/status artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["kernel", "shape", "dtype", "native_gpu_us", "native_cpu_us",
          "raised_resident_us", "ratio_raised_over_native", "comparability",
          "raised_status", "status_detail"]
RESULT = re.compile(r"RESULT kernel=(\S+).*?errors=(\d+)")
UNSAFE = {
    "linalg": r"\blinalg\.", "scf_loop": r"\bscf\.(?:for|while)\b",
    "affine_loop": r"\baffine\.(?:for|parallel)\b",
    "affine_access": r"\baffine\.(?:load|store)\b",
    "memref_copy": r"\bmemref\.copy\b",
    "memref_access": r"\bmemref\.(?:load|store)\b",
}


def keyed(path: Path) -> dict[str, dict[str, str]]:
    return {r["kernel"]: r for r in csv.DictReader(path.open())}


def shape_key(text: str) -> tuple[str, ...]:
    return tuple(sorted(text.replace(" ", "_").split("_")))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", action="append", default=[])
    p.add_argument("--log-dir", action="append", default=[])
    p.add_argument("--output-csv", type=Path, default=ROOT / "aten_benchmark_status.csv")
    p.add_argument("--output-md", type=Path, default=ROOT / "ATEN_BENCHMARK_STATUS.md")
    a = p.parse_args()

    specs = json.loads((ROOT / "resident_shape_specs.json").read_text())
    gpu = keyed(ROOT / "torch_aten_resident_sync_wall.csv")
    cpu = keyed(ROOT / "torch_aten_cpu_sync_wall.csv")
    resident = keyed(ROOT / "resident_silicon.csv")
    provenance = keyed(ROOT / "torch_aten_baseline_provenance.csv")

    built: dict[str, tuple[bool, str]] = {}
    failures: dict[str, str] = {}
    for manifest_name in a.manifest:
        manifest_path = Path(manifest_name)
        data = json.loads(manifest_path.read_text())
        for case in data.get("cases", []):
            abi = manifest_path.parent / case["kernel"] / "artifacts" / "abi_canon.mlir"
            if not abi.exists():
                continue
            text = abi.read_text()
            residual = [name for name, pattern in UNSAFE.items()
                        if re.search(pattern, text)]
            calls = sorted(set(re.findall(
                r"call\s+@(polygeist_(?!(?:cublas_pipeline_(?:begin|end)))[\w]+)", text)))
            safe = bool(calls) and not residual
            detail = (", ".join(calls) if safe else
                      "residual device-unsafe IR: " + ", ".join(residual))
            previous = built.get(case["kernel"])
            if previous is None or (safe and not previous[0]):
                built[case["kernel"]] = (safe, detail)
        for failure in data.get("failures", []):
            failures[failure["kernel"]] = failure["error"]

    runtime: dict[str, tuple[str, str]] = {}
    for directory in a.log_dir:
        for log in Path(directory).glob("*.log"):
            text = log.read_text(errors="replace")
            matches = list(RESULT.finditer(text))
            if matches:
                for match in matches:
                    runtime[match[1]] = ("pass" if match[2] == "0" else "fail",
                                         f"errors={match[2]}")
            else:
                kernel = next((k for k in built if k in log.name), "")
                if kernel and ("Segmentation fault" in text or "exit code: 255" in text):
                    runtime[kernel] = ("fail", "runtime crash or timeout")

    rows = []
    for spec in specs:
        kernel = spec["kernel"]
        g, c, r, prov = gpu[kernel], cpu[kernel], resident.get(kernel), provenance[kernel]
        verified = bool(r and r.get("correctness_scope") ==
                        "device_pointer_output_vs_C_reference")
        comparable = (verified and prov["legal_ratio"] == "yes" and
                      shape_key(r["shape"]) == shape_key(g["shape"]))
        if verified:
            status, detail = "VERIFIED_RESIDENT", "device output matches C reference"
        elif r:
            status, detail = "LEGACY_RESIDENT", "resident timing exists; device output needs strict recheck"
        elif kernel in built:
            if built[kernel][0]:
                if runtime.get(kernel, ("", ""))[0] == "fail":
                    status, detail = "SEMANTIC_OR_RUNTIME_FAILURE", runtime[kernel][1]
                else:
                    status, detail = "DEVICE_SAFE_UNMEASURED", built[kernel][1]
            else:
                status, detail = "RESIDUAL_IR_BLOCKED", built[kernel][1]
        elif kernel in failures:
            status, detail = "BUILD_OR_LOWERING_BLOCKED", failures[kernel]
        else:
            status, detail = "NO_COMPLETE_CURRENT_LIBRARY_REWRITE", "no fresh device-safe whole rewrite"
        raised = float(r["resident_us"]) if r and r.get("resident_us") else None
        native = float(g["time_us"])
        rows.append({
            "kernel": kernel, "shape": spec["shape"], "dtype": spec["dtype"],
            "native_gpu_us": g["time_us"], "native_cpu_us": c["time_us"],
            "raised_resident_us": f"{raised:.6f}" if raised is not None else "",
            "ratio_raised_over_native": f"{raised/native:.6f}" if comparable else "",
            "comparability": prov["comparability"], "raised_status": status,
            "status_detail": detail,
        })

    with a.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    counts = {s: sum(r["raised_status"] == s for r in rows)
              for s in sorted({r["raised_status"] for r in rows})}
    ratios = [float(r["ratio_raised_over_native"]) for r in rows
              if r["ratio_raised_over_native"]]
    missing = [r["kernel"] for r in rows if not r["raised_resident_us"]]
    lines = [
        "# ATen benchmark campaign status", "",
        "All 116 native GPU and x86 CPU baselines use the exact recorded shape/dtype, "
        "five warmups, and best-of-20 synchronized wall time. Raised ratios are emitted "
        "only after a device-pointer output comparison against the extracted C reference.", "",
        f"- Native GPU baselines: {sum(gpu[k['kernel']]['status']=='PASS' for k in specs)}/116",
        f"- Native x86 CPU baselines: {sum(cpu[k['kernel']]['status']=='PASS' for k in specs)}/116",
        f"- Raised resident values: {sum(bool(r['raised_resident_us']) for r in rows)}/116",
        f"- Strictly device-verified raised values: {counts.get('VERIFIED_RESIDENT', 0)}/116",
        f"- Legally comparable raised/native ratios: {len(ratios)}/116",
        f"- Median raised/native ratio (eligible rows only): {statistics.median(ratios):.3f}x" if ratios else
        "- Median raised/native ratio: unavailable", "", "## Raised status", "",
    ]
    lines += [f"- {name}: {count}" for name, count in counts.items()]
    lines += ["", "## Non-verified breakdown", ""]
    for name in sorted(counts):
        if name == "VERIFIED_RESIDENT":
            continue
        members = [r["kernel"] for r in rows if r["raised_status"] == name]
        lines += [f"- {name}: {', '.join(members)}"]
    lines += ["", "## Missing raised resident values", "", ", ".join(missing) or "None", "",
              "The CSV is the authoritative per-kernel artifact. Legacy resident values are "
              "shown for completeness but are not used for paper ratios until rerun through "
              "the strict device-output gate."]
    a.output_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {a.output_csv} and {a.output_md}")


if __name__ == "__main__":
    main()
