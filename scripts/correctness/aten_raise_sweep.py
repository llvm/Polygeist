#!/usr/bin/env python3
"""Measure direct ATen C/C++ translation-unit raising through the PVA pipeline."""

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


EXCLUDED_COMPONENTS = {
    "cuda", "cudnn", "hip", "miopen", "mps", "metal", "vulkan",
    "mkldnn", "mkl", "xnnpack", "kleidiai", "quantized", "ao_sparse", "xpu",
}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
LOOP_RE = re.compile(r"\b(?:affine|scf)\.(?:for|parallel|while)\b")


def discover(native_root: Path):
    return sorted(
        path for path in native_root.rglob("*")
        if path.suffix.lower() in SOURCE_SUFFIXES
        and not (set(path.relative_to(native_root).parts[:-1]) & EXCLUDED_COMPONENTS)
    )


def run(command, timeout, cwd):
    try:
        completed = subprocess.run(
            command, cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout,
            preexec_fn=lambda: os.setsid(),
        )
        return completed.returncode, completed.stdout, completed.stderr, False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return 124, stdout, stderr, True


def concise_error(stderr):
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if "error:" in line or "Assertion" in line or "not handled" in line:
            return line[:500]
    return (lines[0] if lines else "")[:500]


def process_one(source, args):
    started = time.monotonic()
    rel = source.relative_to(args.pytorch)
    with tempfile.TemporaryDirectory(prefix="aten-raise-") as temp:
        lifted = Path(temp) / "lifted.mlir"
        raised = Path(temp) / "raised.mlir"
        compile_command = [
            str(args.cgeist), str(source), "--function=*",
            f"--resource-dir={args.resource_dir}",
            f"-I{args.generated}", f"-I{args.pytorch}",
            f"-I{args.pytorch / 'aten/src'}",
            f"-I{args.pytorch / 'third_party/cpuinfo/include'}",
            "-DC10_USING_CUSTOM_GENERATED_MACROS",
            "-DCPU_CAPABILITY=DEFAULT", "-DCPU_CAPABILITY_DEFAULT",
            "-std=c++20", "--raise-scf-to-affine", "-S", "-o", str(lifted),
        ]
        status, _, stderr, timed_out = run(compile_command, args.timeout, args.root)
        lifted_text = lifted.read_text(errors="replace") if lifted.exists() else ""
        emitted = "func.func" in lifted_text
        input_loops = len(LOOP_RE.findall(lifted_text))
        row = {
            "source": str(rel), "frontend_status": status,
            "frontend_timeout": timed_out, "frontend_emitted": emitted,
            "input_loops": input_loops, "pipeline_status": "",
            "pipeline_timeout": False, "linalg_ops": 0,
            "residual_loops": 0, "raised_any": False, "fully_raised": False,
            "seconds": 0.0, "error": concise_error(stderr),
        }
        if status == 0 and emitted:
            pipeline_command = [
                str(args.opt), str(lifted), "--remove-iter-args",
                "--affine-parallelize", "--raise-affine-to-linalg-pipeline",
                "--lower-polygeist-submap", "-o", str(raised),
            ]
            pstatus, _, pstderr, ptimeout = run(
                pipeline_command, args.timeout, args.root
            )
            raised_text = raised.read_text(errors="replace") if raised.exists() else ""
            linalg_ops = raised_text.count("linalg.")
            residual_loops = len(LOOP_RE.findall(raised_text))
            row.update({
                "pipeline_status": pstatus, "pipeline_timeout": ptimeout,
                "linalg_ops": linalg_ops, "residual_loops": residual_loops,
                "raised_any": input_loops > 0 and linalg_ops > 0,
                "fully_raised": input_loops > 0 and linalg_ops > 0
                                and residual_loops == 0,
                "error": concise_error(pstderr) if pstatus else row["error"],
            })
        row["seconds"] = round(time.monotonic() - started, 3)
        return row


def summarize(rows):
    return {
        "translation_units": len(rows),
        "frontend_success": sum(row["frontend_status"] == 0 for row in rows),
        "frontend_emitted": sum(row["frontend_emitted"] for row in rows),
        "with_loops": sum(row["input_loops"] > 0 for row in rows),
        "pipeline_success": sum(row["pipeline_status"] == 0 for row in rows),
        "raised_any": sum(row["raised_any"] for row in rows),
        "fully_raised": sum(row["fully_raised"] for row in rows),
        "frontend_timeouts": sum(row["frontend_timeout"] for row in rows),
        "pipeline_timeouts": sum(row["pipeline_timeout"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pytorch", type=Path)
    parser.add_argument("--generated", type=Path,
                        default=Path("/tmp/pytorch_aten_codegen"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    opts.root = opts.root.resolve()
    opts.pytorch = (opts.pytorch or opts.root / "third_party/pytorch").resolve()
    opts.generated = opts.generated.resolve()
    opts.cgeist = opts.root / "build/bin/cgeist"
    opts.opt = opts.root / "build/bin/polygeist-opt"
    opts.resource_dir = opts.root / "llvm-project/build/lib/clang/18"
    output = (opts.output or opts.root / "notes/polygeist_raise_to_linalg/aten_raise_sweep_2026_07_21").resolve()
    output.mkdir(parents=True, exist_ok=True)

    sources = discover(opts.pytorch / "aten/src/ATen/native")
    if opts.limit:
        sources = sources[:opts.limit]
    (output / "sources.txt").write_text(
        "\n".join(str(path.relative_to(opts.pytorch)) for path in sources) + "\n"
    )

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures = {pool.submit(process_one, source, opts): source for source in sources}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows.append(future.result())
            if index % 10 == 0 or index == len(sources):
                print(f"[{index}/{len(sources)}] {summarize(rows)}", flush=True)
                checkpoint = sorted(rows, key=lambda row: row["source"])
                (output / "checkpoint.json").write_text(
                    json.dumps(checkpoint, indent=2) + "\n"
                )
    rows.sort(key=lambda row: row["source"])
    fields = list(rows[0]) if rows else []
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "pytorch_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=opts.pytorch, text=True
        ).strip(),
        "scope": {
            "root": "aten/src/ATen/native",
            "suffixes": sorted(SOURCE_SUFFIXES),
            "excluded_path_components": sorted(EXCLUDED_COMPONENTS),
        },
        "summary": summarize(rows), "results": rows,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
