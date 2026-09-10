# ATen paper-analysis artifacts

Regenerate the 42-kernel GPU comparison and the native-library discovery
tables from the current audited inputs with:

```sh
python3 scripts/correctness/generate_aten_paper_analysis.py
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=issues/aten_c_kernels/paper_analysis \
  issues/aten_c_kernels/paper_analysis/aten_gpu_native_vs_raised_42.tex
python3 scripts/correctness/build_ce_viewer.py --paper-analysis-only
```

The runtime figure uses synchronized resident GPU wall time, five warmups,
five measured iterations, and the minimum of the five samples. Both axes are
logarithmic and report microseconds. The black line is parity; the red dashed
line is the 1.6x competitive boundary. All kernels use the same filled green
marker because the plotted results are treated as correctness-gated.

The companion 142-kernel figure excludes the pure `pointwise`,
`pointwise_formula`, and `pointwise_math` semantic families. It uses every
retained native/raised resident pair, preferring the current campaign: 82 rows
use the current minimum-of-five campaign and 60 use the historical raised
best-of-20 sweep. The
per-row timing source and protocol are recorded in
`aten_gpu_nonpointwise_142.csv`; consequently, this figure is a complete
coverage view rather than a single-protocol performance claim. Its red dotted
lines mark twice and half native runtime (`y=2x` and `y=0.5x`). The downward
perpendicular arrow starts on parity and points into the region where the raised
GPU implementation is faster.

The library-discovery table uses the common 598-fixture audit universe.
Counts refer to kernels, not individual call sites. Native paths include
linked vendor libraries and template-generated CUB, Thrust, or CUTLASS
implementations. Polygeist paths count only complete external-library
rewrites. Conditional native backends are source-supported possibilities;
they do not prove that every benchmark shape selects that route.
