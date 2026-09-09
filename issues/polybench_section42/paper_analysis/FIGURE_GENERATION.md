# Figure generation

The HTML viewer embeds the same PGFPlots PDFs used in the Overleaf paper. Their
mirrored source snapshots are:

```text
issues/polybench_section42/paper_analysis/polybench_cpu_native_vs_raised.tex
issues/polybench_section42/paper_analysis/polybench_gpu_native_vs_raised.tex
issues/polybench_section42/paper_analysis/polybench_cpu_min5.csv
issues/polybench_section42/paper_analysis/polybench_gpu_min5.csv
```

Compile the two standalone TeX sources with `pdflatex`:

```text
cd issues/polybench_section42/paper_analysis
pdflatex -interaction=nonstopmode -halt-on-error polybench_cpu_native_vs_raised.tex
pdflatex -interaction=nonstopmode -halt-on-error polybench_gpu_native_vs_raised.tex
```

Both comparison figures plot the native/raised runtime ratio on a base-2
logarithmic axis. The `1x` line means equal performance; values to its right
are raised-code speedups and values to its left are regressions. This single
continuous scale replaces the former CPU broken-axis/cutoff presentation and
keeps large and small ratios visible without truncation. The CPU-versus-GPU
chart is not a paper figure and is intentionally omitted from the HTML
analysis.
