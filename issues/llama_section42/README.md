# Section 4.2 Llama experiment

This directory contains the reproducible, correctness-gated evaluation for the
paper's Llama graph row. The workload is one token through one FP32 7B-size
layer; it is not full Llama inference or quantized GGUF execution.

## Authoritative artifacts

- `protocol_1x5x5_20260908/performance.csv`: current one-process,
  five-warm-up, five-timed-iteration sweep and per-row publication status.
- `protocol_1x5x5_20260908/RUN_STATUS.md`: current correctness results,
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
`POLYGEIST_SILICON_PROFILE=pva-general`. No compiler runs on the Jetson.

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
