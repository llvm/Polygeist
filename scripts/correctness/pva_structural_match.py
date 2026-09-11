#!/usr/bin/env python3
"""Conservative, name-independent PVA semantic-region candidate detector.

This analysis reports structural candidates only.  It never rewrites IR and a
candidate is not an executable/library match until a typed vendor ABI lowering
and correctness validation exist.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    operation: str
    vendor_symbols: tuple[str, str]
    evidence: tuple[str, ...]
    status: str = "structural-candidate-only"


def _count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL))


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def _generic_with(text: str, required: tuple[str, ...], reductions: int) -> bool:
    for match in re.finditer(r"linalg\.generic\b.*?\n\s*}\s*(?:->[^\n]+)?", text,
                             flags=re.DOTALL):
        body = match.group(0)
        iter_match = re.search(r'iterator_types\s*=\s*\[([^]]*)\]', body)
        if not iter_match or iter_match.group(1).count('"reduction"') != reductions:
            continue
        if all(token in body for token in required):
            return True
    return False


def detect(text: str) -> list[Candidate]:
    out: list[Candidate] = []

    # 3x3 unsigned box sum with exact round-to-nearest normalization.
    if (_generic_with(text, ("arith.extui", "arith.addi", "i8", "i32"), 2)
            and _has(text, r"arith\.constant\s+4\s*:\s*i32")
            and _has(text, r"arith\.constant\s+9\s*:\s*i32")
            and _has(text, r"arith\.divsi")
            and _has(text, r"polygeist\.submap\([^\n]*%c3,\s*%c3\)")):
        out.append(Candidate(
            "BoxFilter3x3U8", ("pvaBoxFilterCreate", "pvaBoxFilterSubmit"),
            ("3x3 submap", "two reduction dimensions", "unsigned sum",
             "(sum+4)/9 normalization")))

    # Separable [1,2,1]x[1,2,1] Gaussian and /16 epilogue.
    if (_generic_with(text, ("arith.extui", "arith.muli", "arith.addi",
                             "i8", "i32"), 2)
            and _count(text, r"arith\.muli") >= 2
            and _has(text, r"arith\.constant\s+8\s*:\s*i32")
            and _has(text, r"arith\.constant\s+4\s*:\s*i32")
            and _has(text, r"arith\.shrsi")
            and _has(text, r"memref\.alloca\(\)\s*:\s*memref<3xi32>")):
        out.append(Candidate(
            "GaussianFilter3x3U8",
            ("pvaGaussianFilterCreate", "pvaGaussianFilterSubmit"),
            ("two 3-element coefficient axes", "weighted 2D reduction",
             "unsigned pixels", "(sum+8)>>4 normalization")))

    # Bilateral: range and spatial squared distances feed exp; weight and
    # weight*sample are accumulated separately and divided in the epilogue.
    if (_generic_with(text, ("math.exp", "arith.uitofp", "arith.subf",
                             "arith.mulf", "arith.addf"), 1)
            and _count(text, r"linalg\.yield[^\n]*f32,\s*f32") >= 1
            and _count(text, r"affine\.for") >= 3
            and _has(text, r"affine\.for\s+%[^ ]+\s*=\s*-1\s+to\s+2")
            and _has(text, r"arith\.divf")
            and _has(text, r"arith\.constant\s+255\s*:\s*i32")):
        out.append(Candidate(
            "BilateralFilter3x3U8",
            ("pvaBilateralFilterCreate", "pvaBilateralFilterSubmit"),
            ("3x3 neighborhood", "spatial and range squared distances",
             "exponential weight", "weight and weighted-value reductions",
             "normalized epilogue")))

    # Dilation is a max reduction over an explicit 3x3 neighborhood.
    if (_generic_with(text, ("arith.extui", "arith.cmpi", "arith.select",
                             "i8"), 2)
            and _has(text, r"polygeist\.submap\([^\n]*%c3,\s*%c3\)")):
        out.append(Candidate(
            "MorphologyDilate3x3U8",
            ("pvaMorphologyCreate", "pvaMorphologySubmit"),
            ("3x3 submap", "two reduction dimensions", "unsigned maximum")))

    # Histogram: zero 256 bins followed by data-dependent increment.
    histogram = ((_has(text, r"tensor<256xi32>|memref<256xi32>") or
                  _has(text, r"(?:affine\.for[^\n]*to\s+256|"
                              r"arith\.constant\s+256\s*:\s*index)"))
                 and _has(text, r"arith\.index_castui[^\n]*i8\s+to\s+index")
                 and _has(text, r"arith\.addi[^\n]*%[^,]+,\s*%c1_i32")
                 and _has(text, r"tensor\.insert[^\n]*\[%[^]]+\]"))
    if histogram:
        out.append(Candidate(
            "ImageHistogramU8",
            ("pvaImageHistogramCreate", "pvaImageHistogramSubmit"),
            ("256-bin zero initialization", "unsigned data-dependent bin",
             "unit bin increment")))

    # Full equalization additionally requires a sequential inclusive CDF and
    # indexed CDF remap. Reject the formerly unsound parallel-CDF form.
    has_scan = _has(
        text, r"affine\.for\s+(%[\w.$-]+)\s*=\s*1\s+to\s+256\s*\{.*?"
              r"affine\.load[^\n]*\[\1\s*-\s*1\].*?affine\.store")
    if (histogram and has_scan and _has(text, r"arith\.constant\s+255\s*:\s*i32")
            and _has(text, r"arith\.divsi")
            and not _has(text, r"iterator_types\s*=\s*\[\"parallel\"\].*?"
                              r"arith\.addi[^\n]*%in[^\n]*%in")):
        out.append(Candidate(
            "HistogramEqualizationU8",
            ("pvaHistogramEqualizationCreate",
             "pvaHistogramEqualizationSubmit"),
            ("256-bin histogram", "sequential inclusive CDF",
             "first-nonzero CDF reduction", "255-scale LUT",
             "unsigned indexed remap")))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    candidates = detect(args.input.read_text())
    if args.json:
        print(json.dumps([asdict(c) for c in candidates], indent=2))
    else:
        for candidate in candidates:
            print(f"{candidate.operation}\t{candidate.status}\t"
                  f"{'/'.join(candidate.vendor_symbols)}\t"
                  f"{'; '.join(candidate.evidence)}")


if __name__ == "__main__":
    main()
