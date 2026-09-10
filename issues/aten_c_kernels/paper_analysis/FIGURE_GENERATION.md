# ATen paper-analysis artifacts

Regenerate the 44-kernel GPU comparison and the native-library discovery
tables from the current audited inputs with:

```sh
python3 scripts/correctness/generate_aten_paper_analysis.py
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=issues/aten_c_kernels/paper_analysis \
  issues/aten_c_kernels/paper_analysis/aten_gpu_native_vs_raised_44.tex
python3 scripts/correctness/build_ce_viewer.py --paper-analysis-only
```

The runtime figure uses synchronized resident GPU wall time, five warmups,
five measured iterations, and the minimum of the five samples. Both axes are
logarithmic and report microseconds. The black line is parity; the red dashed
line is the 1.6x competitive boundary. All kernels use the same filled green
marker because the plotted results are treated as correctness-gated.

The companion 115-kernel figure excludes pure elementwise implementations,
including older audit rows labelled `pointwise_reduction_formula` or
`compound_or_specialized` when their selected implementation is a cuDNN
pointwise graph or cuTENSOR unary transform. It uses every retained
native/raised pair, preferring the current campaign: 58 rows use the current
minimum-of-five campaign and 57 use the historical raised
best-of-20 sweep. The
per-row timing source and protocol are recorded in
`aten_gpu_nonpointwise_115.csv`; consequently, this figure is a complete
coverage view rather than a single-protocol performance claim. Its red dotted
lines mark twice and half native runtime (`y=2x` and `y=0.5x`). The downward
perpendicular arrow starts on parity and points into the region where the raised
GPU implementation is faster.

`aten_gpu_nonpointwise_115_table.tex` is the corresponding supplementary
LaTeX longtable. It contains exactly the same 115 rows, runtimes, ratios, and
current-versus-historical source distinction as the figure CSV.

The library-discovery table uses the common 598-fixture audit universe.
Counts refer to kernels, not individual call sites. Native paths include
linked vendor libraries and template-generated CUB, Thrust, or CUTLASS
implementations. Polygeist paths count only complete external-library
rewrites. Conditional native backends are source-supported possibilities;
they do not prove that every benchmark shape selects that route.
