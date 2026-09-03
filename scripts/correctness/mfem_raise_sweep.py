#!/usr/bin/env python3
"""Run cgeist and the affine-to-Linalg pipeline on extracted MFEM kernels."""

import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues" / "mfem_c_kernels"
RESULTS = CORPUS / "results"
CGEIST = ROOT / "build" / "bin" / "cgeist"
OPT = ROOT / "build" / "bin" / "polygeist-opt"
RESOURCE_DIR = ROOT / "llvm-project" / "build" / "lib" / "clang" / "18"


def run(command, log):
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log.write_text(proc.stdout)
    return proc.returncode


def count(pattern, path):
    if not path.exists():
        return 0
    return len(re.findall(pattern, path.read_text()))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((CORPUS / "manifest.csv").open()))
    summary = []
    for row in rows:
        stem = row["id"] + "__" + row["variant"]
        source = CORPUS / row["source"]
        frontend = RESULTS / (stem + ".frontend.mlir")
        raised = RESULTS / (stem + ".raised.mlir")
        front_log = RESULTS / (stem + ".frontend.log")
        raise_log = RESULTS / (stem + ".raise.log")
        front_rc = run([
            str(CGEIST), str(source), "--function=" + row["function"],
            "--resource-dir=" + str(RESOURCE_DIR), "--raise-scf-to-affine",
            "-S", "-o", str(frontend),
        ], front_log)
        raise_rc = -1
        if front_rc == 0:
            raise_rc = run([
                str(OPT), "--select-func=func-name=" + row["function"],
                "--remove-iter-args", "--affine-parallelize",
                "--raise-affine-to-linalg-pipeline",
                # Preserve submap structure through debufferization.  Lowering
                # scratch views first turns them into subviews that hide the
                # producer/consumer tensor graph from the debufferizer.
                str(frontend), "-o", str(raised),
            ], raise_log)
        else:
            raise_log.write_text("not run: cgeist frontend failed\n")
        linalg = count(r"\blinalg\.", raised) if raise_rc == 0 else 0
        loops = count(r"\b(?:affine|scf)\.(?:for|parallel|while)\b", raised) \
            if raise_rc == 0 else 0
        result = dict(row)
        result.update(frontend_ok=str(front_rc == 0).lower(),
                      raise_ok=str(raise_rc == 0).lower(),
                      linalg_ops=str(linalg), residual_loops=str(loops),
                      fully_raised=str(raise_rc == 0 and linalg > 0 and loops == 0).lower())
        summary.append(result)
        print(f"{row['id']:<24} frontend={front_rc == 0!s:<5} "
              f"raise={raise_rc == 0!s:<5} linalg={linalg:<3} loops={loops}")
    fields = list(summary[0]) if summary else []
    with (RESULTS / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
