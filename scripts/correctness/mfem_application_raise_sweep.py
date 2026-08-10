#!/usr/bin/env python3
"""Raise and library-match concrete hot paths from larger MFEM applications."""

import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues" / "mfem_c_kernels" / "application_extractions"
RESULTS = CORPUS / "results"
CGEIST = ROOT / "build" / "bin" / "cgeist"
OPT = ROOT / "build" / "bin" / "polygeist-opt"
MATCHER = ROOT / "scripts" / "correctness" / "kernel_match_rewrite.py"
RESOURCE = ROOT / "llvm-project" / "build" / "lib" / "clang" / "18"


def run(command, output=None, timeout=300):
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + "\ntimeout\n"
    if output is not None and proc.returncode == 0:
        output.write_text(proc.stdout)
    return proc.returncode, proc.stdout, proc.stderr


def count(path, pattern):
    return len(re.findall(pattern, path.read_text())) if path.exists() else 0


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((CORPUS / "manifest.csv").open()))
    summary = []
    for row in rows:
        fn = row["function"]
        source = CORPUS / row["source"]
        front = RESULTS / f"{fn}.frontend.mlir"
        raised = RESULTS / f"{fn}.raised.mlir"
        debuf = RESULTS / f"{fn}.debufferized.mlir"
        matched = RESULTS / f"{fn}.matched.mlir"
        log = RESULTS / f"{fn}.log"
        messages = []

        frc, _, err = run([
            str(CGEIST), str(source), f"--function={fn}",
            f"--resource-dir={RESOURCE}", "--raise-scf-to-affine", "-S",
            "-o", str(front),
        ])
        messages.append("[frontend]\n" + err)
        rrc = drc = mrc = -1
        report = ""
        if frc == 0:
            rrc, _, err = run([
                str(OPT), f"--select-func=func-name={fn}",
                "--remove-iter-args", "--affine-parallelize",
                "--raise-affine-to-linalg-pipeline",
                "--lower-polygeist-submap", str(front), "-o", str(raised),
            ])
            messages.append("[raise]\n" + err)
        raised_loops = count(
            raised, r"\b(?:affine|scf)\.(?:for|parallel|while)\b"
        ) if rrc == 0 else 0
        if rrc == 0 and raised_loops == 0:
            drc, _, err = run([
                str(OPT), "--linalg-debufferize=use-multi-root=true",
                str(raised),
                "-o", str(debuf),
            ])
            messages.append("[debufferize]\n" + err)
        elif rrc == 0:
            messages.append(
                "[debufferize]\nskipped: raised IR still contains "
                f"{raised_loops} residual loop(s)\n"
            )
        if drc == 0:
            mrc, out, err = run([
                "/usr/bin/python3", str(MATCHER), str(debuf), "--dry-run",
            ])
            report = out + err
            messages.append("[matcher]\n" + report)
            if mrc == 0:
                wrc, out, err = run([
                    "/usr/bin/python3", str(MATCHER), str(debuf),
                ], matched)
                messages.append("[rewrite]\n" + err)
                if wrc != 0:
                    mrc = wrc
        log.write_text("\n".join(messages))

        total = re.search(r"total:\s+(\d+) matched / (\d+) bodies", report)
        result = dict(row)
        result.update(
            frontend_ok=str(frc == 0).lower(),
            raise_ok=str(rrc == 0).lower(),
            debufferize_ok=str(drc == 0).lower(),
            matcher_ok=str(mrc == 0).lower(),
            linalg_ops=str(count(raised, r"\blinalg\.")),
            residual_loops=str(raised_loops),
            matched_groups=total.group(1) if total else "0",
            matcher_bodies=total.group(2) if total else "0",
            launches=str(count(matched, r"\bkernel\.launch\b")),
        )
        summary.append(result)
        print(f"{fn:<48} front={frc == 0!s:<5} raise={rrc == 0!s:<5} "
              f"linalg={result['linalg_ops']:<3} loops={result['residual_loops']:<3} "
              f"matches={result['matched_groups']}", flush=True)

    with (RESULTS / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
