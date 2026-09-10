#!/usr/bin/env python3
"""Compile-only raising and library-matching survey for 22 MFEM PA families.

This script intentionally stops at matched MLIR.  It does not lower to an
object, link a runtime, or execute a host/silicon correctness harness.
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues" / "mfem_c_kernels"
SURVEY = CORPUS / "pa_family_compile"
RESULTS = SURVEY / "results"
CGEIST = ROOT / "build" / "bin" / "cgeist"
OPT = ROOT / "build" / "bin" / "polygeist-opt"
MATCHER = ROOT / "scripts" / "correctness" / "kernel_match_rewrite.py"
RESOURCE = ROOT / "llvm-project" / "build" / "lib" / "clang" / "18"
KERNEL_LIBRARY = ROOT / "generic_solver" / "kernel_library_phase2.mlir"

LOOP_RE = re.compile(r"\b(?:affine|scf)\.(?:for|parallel|while)\b")
MATCH_RE = re.compile(r"total:\s+(\d+) matched / (\d+) bodies")
LAUNCH_RE = re.compile(r"kernel\.launch\s+@([A-Za-z0-9_.$-]+)")


def run(command: list[str], timeout: int = 300) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + "\ntimeout\n"


def count(pattern: re.Pattern[str] | str, path: Path) -> int:
    if not path.exists():
        return 0
    return len(re.findall(pattern, path.read_text()))


def first_error(messages: list[str]) -> str:
    for message in messages:
        for line in message.splitlines():
            if "error:" in line.lower() or "timeout" in line.lower():
                return line.strip()[:500]
    return ""


def inject_kernel_definitions(module_text: str) -> str:
    library = KERNEL_LIBRARY.read_text()
    start = library.find("module {")
    if start < 0:
        raise RuntimeError("kernel library has no outer module")
    body_start = library.find("\n", start) + 1
    body_end = library.rfind("\n}")
    definitions = library[body_start:body_end]
    module_match = re.search(r"(?m)^module(?:\s+attributes\s+.*)?\s*\{\s*$",
                             module_text)
    if module_match is None:
        raise RuntimeError("matched IR has no one-line outer module header")
    module_line_end = module_text.find("\n", module_match.start())
    if module_line_end < 0:
        raise RuntimeError("matched IR module header has no body")
    return (module_text[:module_line_end + 1] + definitions + "\n" +
            module_text[module_line_end + 1:])


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((SURVEY / "manifest.csv").open()))
    summary: list[dict[str, str]] = []

    for row in rows:
        family = row["family_id"]
        function = row["function"]
        source = CORPUS / row["source"]
        directory = RESULTS / family
        directory.mkdir(parents=True, exist_ok=True)
        frontend = directory / "frontend.mlir"
        raised = directory / "raised.mlir"
        debufferized = directory / "debufferized.mlir"
        matched = directory / "matched.mlir"
        with_defns = directory / "with_defns.mlir"
        abi = directory / "abi.mlir"
        log = directory / "compile.log"
        messages: list[str] = []

        frc, out, err = run([
            str(CGEIST), str(source), f"--function={function}",
            f"--resource-dir={RESOURCE}", "--raise-scf-to-affine", "-S",
            "-o", str(frontend),
        ])
        messages.append("[frontend]\n" + out + err)

        rrc = drc = mrc = wrc = arc = -1
        report = ""
        if frc == 0:
            rrc, out, err = run([
                str(OPT), f"--select-func=func-name={function}",
                "--remove-iter-args", "--affine-parallelize",
                "--raise-affine-to-linalg-pipeline", str(frontend),
                "-o", str(raised),
            ])
            messages.append("[raising]\n" + out + err)

        # Attempt debufferization even when residual loops remain.  A mixed
        # result is valuable evidence: the matcher may still capture raised
        # subregions while other loops remain explicit.
        if rrc == 0:
            drc, out, err = run([
                str(OPT), "--linalg-debufferize=use-multi-root=true",
                str(raised), "-o", str(debufferized),
            ])
            messages.append("[debufferize]\n" + out + err)

        if drc == 0:
            mrc, out, err = run([
                "/usr/bin/python3", str(MATCHER), str(debufferized),
                "--dry-run",
            ])
            report = out + err
            messages.append("[matcher-dry-run]\n" + report)
            if mrc == 0:
                wrc, rewritten, err = run([
                    "/usr/bin/python3", str(MATCHER), str(debufferized),
                ])
                messages.append("[matcher-rewrite]\n" + err)
                if wrc == 0:
                    matched.write_text(rewritten.rstrip() + "\n")
                    with_defns.write_text(inject_kernel_definitions(rewritten))
                    arc, out, err = run([
                        str(OPT), "--lower-kernel-launch-to-cublas",
                        str(with_defns), "-o", str(abi),
                    ])
                    messages.append("[external-abi-lowering]\n" + out + err)

        total = MATCH_RE.search(report)
        groups = int(total.group(1)) if total else 0
        bodies = int(total.group(2)) if total else 0
        launches = LAUNCH_RE.findall(matched.read_text()) if matched.exists() else []
        raised_loops = count(LOOP_RE, raised)
        result = dict(row)
        result.update({
            "frontend_ok": str(frc == 0).lower(),
            "raise_ok": str(rrc == 0).lower(),
            "debufferize_ok": str(drc == 0).lower(),
            "matcher_ok": str(mrc == 0 and wrc == 0).lower(),
            "linalg_ops": str(count(r"\blinalg\.", raised)),
            "residual_loops": str(raised_loops),
            "fully_raised": str(rrc == 0 and raised_loops == 0).lower(),
            "matcher_bodies": str(bodies),
            "matched_groups": str(groups),
            "kernel_launches": str(len(launches)),
            "launch_symbols": ",".join(sorted(set(launches))),
            "abi_lower_ok": str(arc == 0).lower(),
            "abi_runtime_calls": str(count(r"\bfunc\.call @polygeist_", abi)),
            "abi_residual_launches": str(count(r"\bkernel\.launch\b", abi)),
            "error": first_error(messages),
        })
        summary.append(result)
        log.write_text("\n".join(messages))
        print(f"{family:<24} front={frc == 0!s:<5} raise={rrc == 0!s:<5} "
              f"linalg={result['linalg_ops']:<4} loops={raised_loops:<3} "
              f"matches={groups:<3} launches={len(launches):<3} "
              f"abi={arc == 0!s:<5}", flush=True)

    fields = list(summary[0]) if summary else []
    with (RESULTS / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    captured = [row for row in summary if int(row["kernel_launches"]) > 0]
    abi_captured = [row for row in captured if row["abi_lower_ok"] == "true"]
    invalid_captured = [row for row in captured if row["abi_lower_ok"] != "true"]
    fully_raised = [row for row in summary if row["fully_raised"] == "true"]
    with (RESULTS / "SUMMARY.md").open("w") as out:
        out.write("# MFEM 22-family compile-only coverage\n\n")
        out.write("This survey stops after external-library ABI lowering; "
                  "nothing was linked or executed.\n")
        out.write("Compile-only structural extracts are not correctness evidence.\n\n")
        out.write(f"- families: {len(summary)}\n")
        out.write(f"- frontend successes: {sum(r['frontend_ok']=='true' for r in summary)}\n")
        out.write(f"- raising successes: {sum(r['raise_ok']=='true' for r in summary)}\n")
        out.write(f"- fully raised families: {len(fully_raised)}\n")
        out.write(f"- families with matcher-proposed captures: {len(captured)}\n")
        out.write(f"- families with ABI-valid external-library captures: {len(abi_captured)}\n")
        out.write(f"- families with type/form-invalid proposed captures: {len(invalid_captured)}\n")
        out.write(f"- families with no captures: {len(summary)-len(captured)}\n")
        out.write(f"- matched groups: {sum(int(r['matched_groups']) for r in summary)}\n")
        out.write(f"- emitted kernel.launch operations: {sum(int(r['kernel_launches']) for r in summary)}\n")
        out.write(f"- ABI-valid emitted launches: "
                  f"{sum(int(r['kernel_launches']) for r in abi_captured)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
