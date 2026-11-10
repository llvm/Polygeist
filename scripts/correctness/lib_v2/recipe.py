#!/usr/bin/env python3
"""Single source of truth for the lib_v2 raising recipe.

Library specs (generic_solver/lib_v2/ref_c) and application code (PolyBench)
are raised by *this* module so that both sides see byte-identical pipeline
flags.  The flags are copied from scripts/correctness/polygeist_build.sh,
steps 1-2, default production configuration:

  * non-whole-program, harness == input  -> select-func with
    externalize-dependencies=true
  * DEBUFFERIZE=1, POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE unset (=0) ->
    polygeist.submap views are preserved
  * POLYGEIST_DEBUFFERIZE_MULTI_ROOT unset (=1) ->
    --linalg-debufferize=use-multi-root=true

polygeist_build.sh runs raise + debufferize in one polygeist-opt process; we
run the identical pass list split into two processes so the memref-Linalg
stage can be kept as an artifact (`*_linalg.mlir`).  The split is applied to
both sides.

The alternative "campaign" recipe (scripts/correctness/run_kernel_e2e.sh
--debuf, which produced issues/polybench_section42/ir/<k>/
raised_debufferized.mlir) is kept selectable for comparison only.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# POLYGEIST_BIN_DIR overrides the tool directory (e.g. a side-by-side
# build-v2/bin used to A/B a pass change without touching build/).
BUILD_BIN = Path(os.environ.get("POLYGEIST_BIN_DIR") or (REPO / "build" / "bin"))
CGEIST = str(BUILD_BIN / "cgeist")
POLYGEIST_OPT = str(BUILD_BIN / "polygeist-opt")
RESOURCE_DIR = "/usr/lib/clang/14"

# Optimisation flag passed through to cgeist by the PolyBench campaign
# (POLYBENCH_COMPILE_OPT / polygeist_build.sh gcc passthrough).  Specs get the
# same flag so that cgeist's front-end canonicalisation is identical.
COMMON_CFLAGS = ["-O3"]


@dataclass
class Recipe:
    name: str
    cgeist_tail: list[str]
    raise_passes: list[str]          # after --select-func
    debuf_passes: list[str]
    select_extra: str = ""
    # campaign recipe only: if submaps survive, the campaign lowered to
    # loops instead of debufferizing; we just record that.
    notes: list[str] = field(default_factory=list)

    def cgeist_cmd(self, src: str, fn: str, out: str, cflags: list[str]) -> list[str]:
        return [CGEIST, src, f"--function={fn}", f"--resource-dir={RESOURCE_DIR}",
                *cflags, *self.cgeist_tail, "-o", out]

    def raise_cmd(self, inp: str, fn: str, out: str) -> list[str]:
        sel = f"func-name={fn}" + (f" {self.select_extra}" if self.select_extra else "")
        return [POLYGEIST_OPT, f"--select-func={sel}", *self.raise_passes, inp, "-o", out]

    def debuf_cmd(self, inp: str, out: str) -> list[str]:
        return [POLYGEIST_OPT, *self.debuf_passes, inp, "-o", out]


PRODUCTION = Recipe(
    name="production",
    cgeist_tail=["--raise-scf-to-affine", "-fPIC", "-S"],
    # --delinearize-memref-accesses recovers n-D views of hand-linearized
    # buffers (A[i*S + j]) before dependence analysis; a no-op on PolyBench.
    raise_passes=["--delinearize-memref-accesses", "--remove-iter-args", "--affine-parallelize",
                  "--raise-affine-to-linalg-pipeline"],
    debuf_passes=["--linalg-debufferize=use-multi-root=true"],
    select_extra="externalize-dependencies=true",
)

CAMPAIGN = Recipe(
    name="campaign",
    cgeist_tail=["--raise-scf-to-affine", "-S"],
    raise_passes=["--remove-iter-args", "--affine-parallelize",
                  "--raise-affine-to-linalg-pipeline", "--lower-polygeist-submap"],
    debuf_passes=["--linalg-debufferize"],
    notes=["run_kernel_e2e.sh --debuf; then sed adds `restrict` to to_tensor"],
)

RECIPES = {"production": PRODUCTION, "campaign": CAMPAIGN}


def has_function(mlir_path: str, fn: str) -> bool:
    try:
        text = Path(mlir_path).read_text()
    except OSError:
        return False
    return any("func.func" in ln and f"@{fn}(" in ln for ln in text.splitlines())


@dataclass
class StepResult:
    ok: bool
    rc: int
    cmd: list[str]
    stderr: str
    timed_out: bool = False


def run(cmd: list[str], timeout: int) -> StepResult:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return StepResult(p.returncode == 0, p.returncode, cmd, p.stderr[-4000:])
    except subprocess.TimeoutExpired as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return StepResult(False, -9, cmd, err[-4000:], timed_out=True)


def raise_c(recipe: Recipe, src: str, fn: str, out_prefix: str, cflags: list[str],
            timeout: int = 300) -> dict:
    """Run cgeist -> raise -> debufferize.  Returns a dict of StepResults and
    output paths: <out_prefix>.mlir, <out_prefix>_linalg.mlir,
    <out_prefix>_debuf.mlir (README layout)."""
    affine = f"{out_prefix}.mlir"
    linalg = f"{out_prefix}_linalg.mlir"
    debuf = f"{out_prefix}_debuf.mlir"
    res: dict = {"paths": {"affine": affine, "linalg": linalg, "debuf": debuf}}
    flags = [*COMMON_CFLAGS, *cflags]
    r = run(recipe.cgeist_cmd(src, fn, affine, flags), timeout)
    if r.ok and not has_function(affine, fn):
        # polygeist_build.sh: retry with static-linkage exposure.
        r = run(recipe.cgeist_cmd(src, fn, affine, [*flags, "-Dstatic="]), timeout)
        if r.ok and not has_function(affine, fn):
            r = StepResult(False, 1, r.cmd, f"cgeist did not emit @{fn}")
    res["cgeist"] = r
    if not r.ok:
        return res
    r = run(recipe.raise_cmd(affine, fn, linalg), timeout)
    res["raise"] = r
    if not r.ok:
        return res
    r = run(recipe.debuf_cmd(linalg, debuf), timeout)
    res["debuf"] = r
    if r.ok and recipe is CAMPAIGN:
        t = Path(debuf).read_text()
        import re
        t = re.sub(r"bufferization\.to_tensor (%[^ ]*) :", r"bufferization.to_tensor \1 restrict :", t)
        Path(debuf).write_text(t)
    return res


def shell_line(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
