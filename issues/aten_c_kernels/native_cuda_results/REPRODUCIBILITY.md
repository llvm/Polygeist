# ATen benchmark reproducibility

## Measurement contract

- The primary paper comparison is raised device-resident execution versus
  PyTorch CUDA device-resident execution. Historical mapped-host timings are
  ABI diagnostics only and must not feed headline ratios or the slowness page.
- Shapes and dtypes come only from `resident_shape_specs.json`. The 292-case
  resolved ledger contains all 271 kernels classified as complete mappings to
  a genuine library/runtime definition, plus 21 retained diagnostic or
  partial-match cases from the earlier campaign.
- GPU timings use five warmups and the best of 20 synchronized wall-clock
  measurements. Inputs and outputs remain in `cudaMalloc` storage during the
  timed region; allocation and transfers are excluded.
- Raised timings are publishable only when the same device-pointer invocation
  is copied back and agrees with the extracted C reference before timing.
- CPU timings use the same shape and dtype. The viewer reports separate
  columns for the 24-thread x86-64 host and the 12-thread Jetson AGX Orin CPU;
  they are cross-system context and are never folded into GPU speedup ratios.
- Ratios require exact operation comparability, equal shape/dtype, and the
  strict device-output correctness gate. Proxy and extracted-stage baselines
  remain visible but have no ratio.
- A `_cpu` fixture suffix identifies the ATen source implementation or dispatch
  stub used for extraction. It does not select the benchmark device and does
  not imply that the corresponding PyTorch operation lacks CUDA dispatch.

## Benchmark-shape selection

For scalar and regularly scalable fixtures, the generator uniformly scales
the extracted compile-time dimensions until the largest input or output array
contains approximately 4,194,304 elements. For structured operators, explicit
shapes preserve the extracted layout, batch and reduction dimensions, sparse
storage interpretation, output materialization, and the preconditions of the
candidate vendor API while remaining within device memory. Consequently these
are called *benchmark shapes*, not uniformly large or representative production
shapes. The explicit structured cases are the `CASES` entries in
`scripts/correctness/aten_pointwise_graph_silicon.py`; the generated resolved
set is `resident_shape_specs.json`.

Raised, PyTorch CUDA, and both PyTorch CPU measurements consume that same
resolved shape and dtype. CPU numbers are reported as cross-system context
rather than same-hardware CPU/GPU speedups.

The generator resolves the native operation from the exhaustive native audit;
it does not require the operation to appear in an older timing CSV. Every
recipe routed through the explicit native-fixture adapter has a required
per-kernel verdict in `native_fixture_adjudication.csv`. The provenance build
fails if a fixture is missing from that audit or if a stale row remains. Exact
operations and explicitly bounded finite-input benchmark domains may publish a
ratio; invalid recipes, storage/layout proxies, RNG-scope mismatches, and
missing native adapters remain visible without a ratio.

## Pinned environment

- Source revision at campaign start: `5a5bb5e86f64002d22b11917f5a36a41567bf982`
  on branch `raisetolinalg` (the generated status records include the fixes
  made after this revision).
- GPU: Jetson AGX Orin, SM87, CUDA 12.6.
- GPU framework baseline: PyTorch 2.6.0+cu126.
- CPU framework baseline: PyTorch 2.6.0+cpu in isolated `/tmp/tpy_x86`.
- Jetson CPU context: PyTorch 2.8.0+cpu in isolated
  `/home/nvidia/tpy-cpu-site`, using 12 intra-op and 12 inter-op threads. This
  version differs from the x86 and CUDA framework baselines and is labelled
  explicitly in the provenance CSV and viewer.
- cuDNN headers/runtime ABI: 9.22.0.
- cuTENSOR headers/runtime ABI: 2.0.0.
- Companion libraries are staged under
  `/home/nvidia/polygeist_cuda_libs`; the CUB library must export
  `polygeist_cub_segmented_reduce_f64_cuda`.

## Commands

Generate the common shape specification:

```
python3 issues/aten_c_kernels/native_cuda_results/gen_resident_shape_specs.py
```

Run the PyTorch baselines with the isolated environments (the benchmark emits
machine-readable metadata and can be subset with `ATEN_BENCH_KERNELS`):

```
ATEN_BENCH_DEVICE=cuda ATEN_BENCH_TIMING=sync_wall \
  python3 issues/aten_c_kernels/native_cuda_results/bench_shaped.py

ATEN_BENCH_DEVICE=cpu ATEN_BENCH_TIMING=sync_wall \
  /tmp/tpy_x86/bin/python \
  issues/aten_c_kernels/native_cuda_results/bench_shaped.py
```

The Jetson CPU command uses the same script and generated spec file after they
are staged to the device:

```
PYTHONPATH=/home/nvidia/tpy-cpu-site ATEN_BENCH_DEVICE=cpu \
ATEN_BENCH_TIMING=sync_wall python3 bench_shaped.py resident_shape_specs.json
```

Parse logs and regenerate provenance:

```
python3 issues/aten_c_kernels/native_cuda_results/parse_bench_log.py --help
python3 issues/aten_c_kernels/native_cuda_results/gen_native_provenance.py
```

Build explicit raised cases. The compatibility environment preserves the
historical flat-Linalg path while the default exercises composed submaps:

```
POLYGEIST_FORCE_RESIDENT=1 \
  python3 scripts/correctness/aten_pointwise_graph_silicon.py \
  --kernel aten_sum --output /tmp/aten-resident --jobs 1

POLYGEIST_FORCE_RESIDENT=1 \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
  python3 scripts/correctness/aten_pointwise_graph_silicon.py \
  --kernel aten_addmm --output /tmp/aten-resident-flat --jobs 1
```

Run a built executable on Orin using the documented profile; credentials stay
outside the repository:

```
POLYGEIST_SILICON_PROFILE=pva-general POLYGEIST_JETSON_RUNS=3 \
  scripts/correctness/run_jetson.sh --exe \
  /tmp/aten-resident/aten_sum/aten_sum aten_sum_resident
```

Aggregate only correctness-gated logs, then generate the campaign summary:

```
python3 issues/aten_c_kernels/native_cuda_results/aggregate_resident_logs.py \
  '/tmp/aten-logs/*.silicon.log' \
  --existing issues/aten_c_kernels/native_cuda_results/resident_silicon.csv \
  --output issues/aten_c_kernels/native_cuda_results/resident_silicon.csv \
  --date YYYY-MM-DD

python3 issues/aten_c_kernels/native_cuda_results/summarize_campaign.py \
  --manifest /tmp/aten-resident/manifest.json \
  --log-dir /tmp/aten-logs
```

Finally regenerate the ATen viewer with:

```
python3 scripts/correctness/build_ce_viewer.py --aten-only
```
