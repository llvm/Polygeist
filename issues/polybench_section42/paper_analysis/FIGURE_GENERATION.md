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

The related-work reproduction matrix and percentage-speedup figure are
generated with:

```text
python3 generate_related_work_comparison.py
python3 generate_speedup_related_work.py
```

Their x86 Polygeist CPU series comes from
`polybench_cpu_raised_fp32_x86_min5.csv`. mlirSynth remains marked as lacking a
comparable runtime because the released artifact omits the paper's LARGE-FP32
XLA execution harness; its separately reproduced synthesis results must not be
plotted as execution speedups.

The B300 FP32 figure reads all correctness-passing native PolyBenchGPU rows
directly from `../b300_polybenchgpu_fp32_2026-09-09/summary.csv`, rather than a
hand-maintained subset. These native and Polygeist runs used different physical
B300 workers, so the figure marks their comparison as secondary cross-worker
evidence. The x86 figure includes the retained native-correlation baseline and
the common-set `trisolv` row. KernelFaRer `2mm`/`3mm` use the corrected
non-nested two- and three-CBLAS-call executions and pass complete outputs.
