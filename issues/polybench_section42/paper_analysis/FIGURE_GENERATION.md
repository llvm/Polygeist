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

The CPU comparison plots the native/raised runtime ratio on a base-2
logarithmic axis. This single continuous scale replaces its former broken-axis
presentation and keeps large and small ratios visible without truncation. The
GPU comparison retains its linear percentage axis because it did not require a
break; its positive and negative PGFPlots series both use `bar shift=0pt` so
every bar stays vertically centered on its kernel row. The CPU-versus-GPU chart
is not a paper figure and is intentionally omitted from the HTML analysis.
