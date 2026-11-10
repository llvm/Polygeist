#!/usr/bin/env python3
"""Focused structural audit for the generated B300 benchmark page."""

from pathlib import Path


root = Path("/tmp/ir_viewer")
index = (root / "index.html").read_text()
page = (root / "b300.html").read_text()

assert 'href="b300.html"' in index
assert page.count("<tr>") - 3 == 136
for required in (
    "Polygeist FP64",
    "Polygeist FP32",
    "PolyBenchGPU FP32",
    "PPCG",
    "PolyBench-ACC",
    "Ginsbach-regions",
    "KernelFaRer",
    "Pluto",
    "Polly",
):
    assert required in page, required

print("PASS: landing link and 136 B300/related-work data rows")
