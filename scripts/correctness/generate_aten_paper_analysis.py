#!/usr/bin/env python3
"""Generate reproducible ATen paper figures and library-overlap tables."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATEN = ROOT / "issues/aten_c_kernels"
CAMPAIGN = ATEN / "native_cuda_results/section42_campaign/results.csv"
BENCHMARK_STATUS = ATEN / "native_cuda_results/aten_benchmark_status.csv"
RESIDENT_SILICON = ATEN / "native_cuda_results/resident_silicon.csv"
RAISE_SUMMARY = ATEN / "results/summary.tsv"
NATIVE_AUDIT = ATEN / "native_cuda_external_library_calls.csv"
RAISED_AUDIT = ATEN / "cuda_library_audit.csv"
OUTPUT = ATEN / "paper_analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def campaign_value(row: dict[str, str], phase: str) -> str:
    return row.get(f"{phase}_min_us", "") or row.get(
        f"{phase}_median_us", "")


def phase_ok(row: dict[str, str], phase: str) -> bool:
    return bool(campaign_value(row, phase)) and row.get(
        f"{phase}_errors", "") == "0"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def competitive_rows() -> list[dict[str, object]]:
    audit = {row["kernel"]: row for row in read_csv(RAISED_AUDIT)}
    rows = []
    for row in read_csv(CAMPAIGN):
        if row.get("comparability", "").startswith("FRAMEWORK_LEVEL"):
            continue
        if not (phase_ok(row, "native_gpu") and phase_ok(row, "raised")):
            continue
        native = float(campaign_value(row, "native_gpu"))
        raised = float(campaign_value(row, "raised"))
        ratio = raised / native
        if ratio >= 1.6:
            continue
        route = audit.get(row["kernel"], {})
        rows.append({
            "id": len(rows) + 1,
            "kernel": row["kernel"],
            "native_us": native,
            "raised_us": raised,
            "raised_over_native": ratio,
            "speedup_percent": 100.0 * (native / raised - 1.0),
            "semantic_family": route.get("semantic_family", "unclassified"),
            "raised_library": route.get("candidate_library", ""),
            "raised_call": route.get("current_match", ""),
        })
    rows.sort(key=lambda item: (float(item["native_us"]), str(item["kernel"])))
    for index, row in enumerate(rows, 1):
        row["id"] = index
    if len(rows) != 44:
        raise SystemExit(f"competitive cohort drift: expected 44, got {len(rows)}")
    return rows


def write_competitive_csv(rows: list[dict[str, object]]) -> None:
    path = OUTPUT / "aten_gpu_competitive_44.csv"
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nonpointwise_rows() -> list[dict[str, object]]:
    audit = {row["kernel"]: row for row in read_csv(RAISED_AUDIT)}
    status = {row["kernel"]: row for row in read_csv(BENCHMARK_STATUS)}
    campaign = {row["kernel"]: row for row in read_csv(CAMPAIGN)}
    pure_pointwise = {"pointwise", "pointwise_formula", "pointwise_math"}

    def is_pure_pointwise(route: dict[str, str]) -> bool:
        family = route.get("semantic_family", "")
        match = route.get("current_match", "")
        if family in pure_pointwise:
            return True
        # Some formulas predate the pointwise-family split in the audit.  Only
        # exclude these broader labels when the selected implementation itself
        # proves that this is an elementwise graph or unary transform.
        return (family in {"pointwise_reduction_formula",
                           "compound_or_specialized"}
                and (match.startswith("cudnnPointwiseGraph_")
                     or match.startswith("cutensorUnary_")))
    rows = []
    for kernel, route in audit.items():
        if not (route.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE"
                and route.get("counts_as_library_reuse") == "yes"):
            continue
        if is_pure_pointwise(route):
            continue
        current = campaign.get(kernel, {})
        if phase_ok(current, "native_gpu") and phase_ok(current, "raised"):
            native_text = campaign_value(current, "native_gpu")
            raised_text = campaign_value(current, "raised")
            timing_source = "current Section 4.2 campaign"
            timing_protocol = "minimum of 5 after 5 warmups"
        else:
            recorded = status.get(kernel, {})
            native_text = recorded.get("native_gpu_us", "")
            raised_text = recorded.get("raised_resident_us", "")
            if not (native_text and raised_text):
                continue
            timing_source = "historical resident sweep"
            timing_protocol = "raised best of 20; retained native baseline"
        native = float(native_text)
        raised = float(raised_text)
        rows.append({
            "id": 0,
            "kernel": kernel,
            "native_us": native,
            "raised_us": raised,
            "raised_over_native": raised / native,
            "speedup_percent": 100.0 * (native / raised - 1.0),
            "semantic_family": route.get("semantic_family", "unclassified"),
            "raised_library": route.get("candidate_library", ""),
            "raised_call": route.get("current_match", ""),
            "timing_source": timing_source,
            "timing_protocol": timing_protocol,
        })
    rows.sort(key=lambda item: (float(item["native_us"]), str(item["kernel"])))
    for index, row in enumerate(rows, 1):
        row["id"] = index
    if len(rows) != 115:
        raise SystemExit(f"non-pointwise cohort drift: expected 115, got {len(rows)}")
    return rows


def write_nonpointwise_csv(rows: list[dict[str, object]]) -> None:
    path = OUTPUT / "aten_gpu_nonpointwise_115.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_nonpointwise_table_tex(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{longtable}{lrrrl}",
        r"  \caption{Per-kernel timing data for the 115 non-pointwise ATen "
        r"kernels in Figure~\ref{fig:aten-gpu-runtime}. Times are in "
        r"microseconds.}\label{tab:aten-nonpointwise-timings} \\",
        r"  \toprule",
        r"  Kernel & Native CUDA & Raised GPU & Raised/native & Source \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \toprule",
        r"  Kernel & Native CUDA & Raised GPU & Raised/native & Source \\",
        r"  \midrule",
        r"  \endhead",
    ]
    for row in rows:
        source = ("current min-of-five" if row["timing_source"] ==
                  "current Section 4.2 campaign" else "historical fallback")
        lines.append(
            f"  {latex_escape(str(row['kernel']))} & "
            f"{float(row['native_us']):.3f} & {float(row['raised_us']):.3f} & "
            f'{float(row["raised_over_native"]):.3f}$\\times$ & {source} \\\\')
    lines.extend([r"  \bottomrule", r"\end{longtable}", ""])
    (OUTPUT / "aten_gpu_nonpointwise_115_table.tex").write_text(
        "\n".join(lines))


def coordinates(rows: list[dict[str, object]]) -> str:
    return " ".join(
        f"({float(row['native_us']):.6f},{float(row['raised_us']):.6f})"
        for row in rows)


def write_figure_tex(rows: list[dict[str, object]]) -> None:
    points = coordinates(rows)
    tex = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{nativeblue}}{{HTML}}{{0969DA}}
\definecolor{{raisedgreen}}{{HTML}}{{1A7F37}}
\definecolor{{regressionred}}{{HTML}}{{CF222E}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{loglogaxis}}[
  width=6.65in, height=5.15in,
  title={{ATen: raised GPU vs native GPU runtime}},
  title style={{font=\bfseries\normalsize}},
  xlabel={{Native CUDA runtime ($\mu$s)}},
  ylabel={{Raised GPU runtime ($\mu$s)}},
  xmin=70, xmax=10000, ymin=70, ymax=10000,
  grid=both, minor grid style={{draw=gray!12}},
  major grid style={{draw=gray!25}},
  ticklabel style={{font=\small}}, label style={{font=\small}},
  legend style={{font=\scriptsize,at={{(0.02,0.98)}},anchor=north west,
                 fill=white,fill opacity=0.92,draw=gray!35}},
  axis lines*=left,
]
\addplot+[no marks,draw=black,line width=0.9pt]
  coordinates {{(70,70) (10000,10000)}};
\addlegendentry{{parity ($y=x$)}}
\addplot+[no marks,draw=regressionred,dashed,line width=0.9pt]
  coordinates {{(70,112) (6250,10000)}};
\addlegendentry{{competitive bound ($y=1.6x$)}}
\addplot+[only marks,mark=*,mark size=2.4pt,
  draw=raisedgreen!85!black,fill=raisedgreen]
  coordinates {{{points}}};
\addlegendentry{{ATen kernels}}
\end{{loglogaxis}}
\end{{tikzpicture}}
\end{{document}}
"""
    (OUTPUT / "aten_gpu_native_vs_raised_44.tex").write_text(tex)


def write_nonpointwise_figure_tex(rows: list[dict[str, object]]) -> None:
    points = coordinates(rows)
    tex = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{raisedgreen}}{{HTML}}{{1A7F37}}
\definecolor{{regressionred}}{{HTML}}{{CF222E}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{loglogaxis}}[
  width=6.65in, height=5.15in,
  title={{ATen: 115 non-pointwise raised vs native GPU runtimes}},
  title style={{font=\bfseries\normalsize}},
  xlabel={{Native CUDA runtime ($\mu$s)}},
  ylabel={{Raised GPU runtime ($\mu$s)}},
  xmin=50, xmax=10000, ymin=50, ymax=100000,
  grid=both, minor grid style={{draw=gray!12}},
  major grid style={{draw=gray!25}},
  ticklabel style={{font=\small}}, label style={{font=\small}},
  legend style={{font=\scriptsize,at={{(0.02,0.98)}},anchor=north west,
                 fill=white,fill opacity=0.92,draw=gray!35}},
  axis lines*=left,
]
\addplot+[no marks,draw=black,line width=0.9pt]
  coordinates {{(50,50) (10000,10000)}};
\addlegendentry{{parity ($y=x$)}}
\addplot+[no marks,draw=regressionred,densely dotted,line width=1.0pt]
  coordinates {{(50,100) (10000,20000)}};
\addlegendentry{{$2\times$ native time ($y=2x$)}}
\addplot+[no marks,draw=regressionred,densely dotted,line width=1.0pt]
  coordinates {{(100,50) (10000,5000)}};
\addlegendentry{{$0.5\times$ native time ($y=0.5x$)}}
\addplot+[only marks,mark=*,mark size=1.9pt,
  draw=raisedgreen!85!black,fill=raisedgreen]
  coordinates {{{points}}};
\addlegendentry{{non-pointwise ATen kernels}}
\draw[->,black,line width=1.0pt]
  (axis cs:2300,2300) -- (axis cs:2620,1310)
  node[pos=1,below right,font=\scriptsize,fill=white,inner sep=1.5pt]
  {{raised GPU faster}};
\end{{loglogaxis}}
\end{{tikzpicture}}
\end{{document}}
"""
    (OUTPUT / "aten_gpu_native_vs_raised_nonpointwise_115.tex").write_text(tex)


def normalized_raised_library(row: dict[str, str]) -> str:
    library = row.get("candidate_library", "")
    if library == "cuDNN Resample":
        return "cuDNN"
    if library == "cuBLASLt":
        return "cuBLAS"
    return library


def write_library_tables() -> None:
    native = {row["kernel"]: row for row in read_csv(NATIVE_AUDIT)}
    raised = {row["kernel"]: row for row in read_csv(RAISED_AUDIT)}
    kernels = sorted(set(native) & set(raised))

    def native_external(kernel: str) -> bool:
        return native[kernel].get("audit_status") == "EXTERNAL_LIBRARY"

    def raised_external(kernel: str) -> bool:
        row = raised[kernel]
        return (row.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE"
                and row.get("counts_as_library_reuse") == "yes")

    both = [k for k in kernels if native_external(k) and raised_external(k)]
    native_only = [k for k in kernels if native_external(k) and not raised_external(k)]
    raised_only = [k for k in kernels if not native_external(k) and raised_external(k)]
    neither = [k for k in kernels if not native_external(k) and not raised_external(k)]
    same_family, different_family = [], []
    for kernel in both:
        native_families = set(native[kernel]["external_libraries"].split("|"))
        raised_family = normalized_raised_library(raised[kernel])
        (same_family if raised_family in native_families else different_family).append(kernel)
    linked = {"cuBLAS", "cuDNN", "cuSPARSE", "cuSOLVER", "cuFFT"}
    templates = {"CUB", "Thrust", "CUTLASS"}
    native_linked = sum(bool(set(native[k]["external_libraries"].split("|")) & linked)
                        for k in kernels if native_external(k))
    native_templates = sum(bool(set(native[k]["external_libraries"].split("|")) & templates)
                           for k in kernels if native_external(k))
    native_linked_and_templates = sum(
        bool(set(native[k]["external_libraries"].split("|")) & linked)
        and bool(set(native[k]["external_libraries"].split("|")) & templates)
        for k in kernels if native_external(k))
    different_directions = Counter()
    for kernel in different_family:
        native_label = "/".join(native[kernel]["external_libraries"].split("|"))
        different_directions[(native_label,
                              normalized_raised_library(raised[kernel]))] += 1
    summary = [
        ("Fixtures in common audit universe", len(kernels)),
        ("Native ATen paths using an external library", len(both) + len(native_only)),
        ("Native paths using linked cuBLAS/cuDNN/cuSPARSE/etc.", native_linked),
        ("Native paths using CUB/Thrust/CUTLASS templates", native_templates),
        ("Native paths counted in both implementation-form rows",
         native_linked_and_templates),
        ("Complete Polygeist external-library mappings", len(both) + len(raised_only)),
        ("Found by both", len(both)),
        ("Found only in native ATen", len(native_only)),
        ("Found only by Polygeist", len(raised_only)),
        ("Found by neither", len(neither)),
        ("Same library family among common findings", len(same_family)),
        ("Different library family among common findings", len(different_family)),
    ]
    summary.extend(
        (f"Different family: native {native_family} to Polygeist {raised_family}",
         count)
        for (native_family, raised_family), count
        in sorted(different_directions.items())
    )
    with (OUTPUT / "aten_native_library_comparison.csv").open(
            "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "kernel_count"])
        writer.writerows(summary)

    lines = [
        r"\begin{table}[t]", r"  \centering", r"  \small",
        r"  \caption{External-library use in native ATen CUDA implementations versus complete library mappings recovered by \theproject{}. CUB, Thrust, and CUTLASS are counted as template-generated library implementations. Counts are kernels, not call sites, and multi-library native paths are counted once.}",
        r"  \label{tab:aten-library-overlap}",
        r"  \begin{tabular}{lr}", r"    \toprule",
        r"    Classification & Kernels \\", r"    \midrule",
    ]
    for label, count in summary:
        escaped = latex_escape(label).replace(
            r" to Polygeist ", r" $\rightarrow$ Polygeist ")
        lines.append(f"    {escaped} & {count} \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    (OUTPUT / "aten_native_library_comparison.tex").write_text("\n".join(lines))

    detail_fields = ["kernel", "native_libraries", "native_symbol",
                     "raised_library", "raised_call"]
    with (OUTPUT / "aten_different_library_families.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields,
                                lineterminator="\n")
        writer.writeheader()
        for kernel in different_family:
            writer.writerow({
                "kernel": kernel,
                "native_libraries": native[kernel]["external_libraries"],
                "native_symbol": native[kernel]["evidence_symbol"],
                "raised_library": normalized_raised_library(raised[kernel]),
                "raised_call": raised[kernel]["current_match"],
            })


def write_coverage_table() -> None:
    audit = read_csv(RAISED_AUDIT)
    summary = read_tsv(RAISE_SUMMARY)
    complete = {
        row["kernel"] for row in audit
        if row.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE"
        and row.get("counts_as_library_reuse") == "yes"
    }
    partial = sum(row.get("current_match_scope") == "PARTIAL_STAGE_ONLY"
                  for row in audit)
    unmatched = sum(row.get("current_match_scope") == "NONE" for row in audit)

    timed = {
        row["kernel"] for row in read_csv(RESIDENT_SILICON)
        if row.get("resident_us")
    }
    timed.update(
        row["kernel"] for row in read_csv(BENCHMARK_STATUS)
        if row.get("raised_resident_us")
    )
    timed.update(
        row["kernel"] for row in read_csv(CAMPAIGN)
        if phase_ok(row, "raised")
    )
    executed = len(complete & timed)

    fully_linalg = sum(int(row["linalg_ops"]) > 0
                       and int(row["residual_loops"]) == 0
                       for row in summary)
    mixed = sum(int(row["linalg_ops"]) > 0
                and int(row["residual_loops"]) > 0
                for row in summary)
    no_linalg = sum(int(row["linalg_ops"]) == 0 for row in summary)
    total = len(audit)
    assert len(summary) == total == 598
    assert len(complete) + partial + unmatched == total
    assert fully_linalg + mixed + no_linalg == total
    assert executed == 217

    rows = [
        ("Corpus and library execution", "ATen standalone kernel fixtures", total,
         "complete audited fixture corpus"),
        ("Corpus and library execution", "Complete library rewrites", len(complete),
         "complete whole-fixture static library mapping"),
        ("Corpus and library execution", "Complete rewrites executed on GPU", executed,
         "recorded raised resident GPU runtime"),
        ("Corpus and library execution", "Complete rewrites awaiting GPU execution",
         len(complete) - executed,
         "complete static mapping without a resident timing"),
        ("Corpus and library execution", "Partial library matches", partial,
         "library stage found with residual computation"),
        ("Corpus and library execution", "No library match found", unmatched,
         "no current complete or partial library match"),
        ("Raised-IR structure", "Fully raised to Linalg", fully_linalg,
         "one or more Linalg operations and no residual affine/SCF loops"),
        ("Raised-IR structure", "Mixed Linalg and structured loops", mixed,
         "Linalg plus residual affine/SCF loops"),
        ("Raised-IR structure", "No Linalg operation", no_linalg,
         "loop-only or scalar/control-flow raised IR"),
    ]
    with (OUTPUT / "aten_coverage_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["section", "metric", "kernel_count", "definition"])
        writer.writerows(rows)

    lines = [
        r"\begin{table}[t]", r"  \centering", r"  \small",
        r"  \caption{ATen corpus, library-mapping, execution, and raised-IR coverage. Library status and IR structure are separate classifications; the three IR rows partition all 598 fixtures.}",
        r"  \label{tab:aten-coverage-summary}",
        r"  \begin{tabular}{lr}", r"    \toprule",
        r"    Measurement & Kernels \\", r"    \midrule",
    ]
    last_section = ""
    for section, metric, count, _ in rows:
        if section != last_section:
            if last_section:
                lines.append(r"    \addlinespace")
            lines.append(
                rf"    \multicolumn{{2}}{{l}}{{\textit{{{latex_escape(section)}}}}} \\")
            last_section = section
        lines.append(f"    {latex_escape(metric)} & {count} \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    (OUTPUT / "aten_coverage_summary.tex").write_text("\n".join(lines))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = competitive_rows()
    write_competitive_csv(rows)
    write_figure_tex(rows)
    nonpointwise = nonpointwise_rows()
    write_nonpointwise_csv(nonpointwise)
    write_nonpointwise_table_tex(nonpointwise)
    write_nonpointwise_figure_tex(nonpointwise)
    write_library_tables()
    write_coverage_table()
    print(f"wrote {len(rows)} competitive and {len(nonpointwise)} non-pointwise "
          f"rows plus library tables to {OUTPUT}")


if __name__ == "__main__":
    main()
