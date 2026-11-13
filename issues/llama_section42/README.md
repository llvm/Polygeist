# Section 4.2 Llama experiment

This directory contains the reproducible, correctness-gated evaluation for the
paper's Llama graph row. The workload is one token through one FP32 7B-size
layer; it is not full Llama inference or quantized GGUF execution.

## Authoritative artifacts

- `../llama_fixed_20261110/REPORT.txt`: fresh-source repair and retest, with
  `results.json`, all logits, raw samples, and commands in the same directory.
- `protocol_1x5x5_20260908/performance.csv`: historical one-process,
  five-warm-up, five-timed-iteration sweep and per-row publication status.
- `protocol_1x5x5_20260908/RUN_STATUS.md`: historical correctness results,
  provenance qualifications, hashes, and next action.
- `protocol_1x5x5_20260908/full_outputs/`: complete 32,000-logit outputs for
  all four implementations.
- `performance.csv`: older retained measurements. These are historical and
  are not mixed with the current protocol sweep.
- `kernel_provenance.csv`: external-library matches, CPU residuals, excluded
  project-authored helpers, and unsupported source forms.
- `REPRODUCTION_STATUS.md`: scope, revisions, fixes, and current conclusions.
- `logs/llama_*_dump_7b.csv`: complete 32,000-logit outputs.
- `logs/full_logit_comparison.log`: independent tolerance check.

## Rebuild

Everything is cross-compiled on the x86 host. Point
`POLYGEIST_CUTENSOR_ROOT` at an AArch64 cuTENSOR include/lib tree and run:

```sh
POLYGEIST_CUTENSOR_ROOT=/path/to/aarch64/cutensor \
  issues/llama_section42/build_jetson_benchmarks.sh
```

The script produces ignored binaries and ggml build products under this
directory. Deploy them with `scripts/correctness/run_jetson.sh` and
`POLYGEIST_SILICON_PROFILE=configured-host`. No compiler runs on the Jetson.

Recheck full outputs with:

```sh
issues/llama_section42/compare_logits.py \
  issues/llama_section42/logs/llama_native_cpu_dump_7b.csv \
  issues/llama_section42/logs/llama_polygeist_external_only_dump_7b.csv \
  issues/llama_section42/logs/llama_ggml_cuda_dump_7b.csv
```

## Interpretation

The current sweep has two eligible rows: native Orin CPU as the numerical
reference and ggml CUDA as the expert baseline. Both fresh Polygeist rows are
excluded because their complete outputs fail the predeclared strict gate. The
raised timings remain visible as diagnostics but cannot support a speedup
claim. Split RoPE and a branchless mask are explicit source accommodations and
must remain visible in any paper or viewer description.

The paper-facing HTML summary is generated as `llama-paper.html` by
`scripts/correctness/build_ce_viewer.py`. It publishes only correctness-eligible
rows in its headline graphs and keeps failed rows in a separate audit table.

## Source rebuild and numerical audit (2026-11-10)

The fresh-source driver is `scripts/correctness/build_llama_reproduction.sh`.
It builds native CPU, the raised hybrid, the raised resident session, and the
expert ggml harness. Supply `POLYGEIST_CUTENSOR_ROOT` and, if needed,
`POLYGEIST_GGML_BUILD` (the pinned AArch64 ggml build described above):

```sh
POLYGEIST_CUTENSOR_ROOT=/path/to/aarch64/cutensor \
  scripts/correctness/build_llama_reproduction.sh /tmp/llama-reproduction
```

The resident session keeps fixed C array extents, uses compiler-generated GPU
residual kernels, and invokes timing callbacks for five warmups and five
measured forwards. Run it with `POLYGEIST_CUDA_GRAPH=1`; the final retest uses
one captured graph across the complete device sequence. Keep generic cuDNN
pointwise matching enabled, as it was in the retained submission-time code.
`RESIDENT_SAMPLE` columns are iteration, synchronized wall
milliseconds, and CUDA-event milliseconds. Its session total additionally
includes initial allocation/transfers and final copy-back. Argument residency
assumes the harness's independently allocated arrays do not overlap. This
assumption belongs to this harness, not arbitrary C callers.

The expert executable runs with `--warmup 5 --iters 5 --dump-all`. It uses the
real ggml CUDA backend from whisper.cpp revision
`f24588a272ae8e23280d9c220536437164e6ed28`. It is a fixture graph, not an invocation
of the llama.cpp application. Keep that distinction in the artifact and paper.

`compare_llama_outputs.py` checks every indexed logit, rejects missing,
duplicate, and non-finite values, and reports both historical gates for both
GPU implementations. The scalar FP32 CPU result is not an exact oracle:
sequential FP32 reductions accumulate different rounding error from parallel
library reductions. The 2026-11-10 diagnostic also evaluates the unchanged
algorithm with double intermediates and exactly the same FP32-initialized
inputs; array hashes verify that input equality. The strict CPU comparison
remains reported even when the independent higher-precision comparison passes.

The current retest evidence is in `../llama_fixed_20261110/REPORT.txt`, with
complete outputs, compiler IR, build/run commands, and board probes alongside
it. These diagnostic results must retain the automatic-fan qualification;
the canonical publication protocol requires a fixed fan setting.
