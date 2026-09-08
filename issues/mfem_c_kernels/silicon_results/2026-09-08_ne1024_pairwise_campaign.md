# MFEM normalized-kernel NE=1024 timing campaign — 2026-09-08

## Outcome

The clean sequential campaign completed 255/255 executable runs and retained
5,100 individual timing samples: 17 kernels, three implementations, five fresh
processes, and twenty samples per process. Every process performed five warmup
calls first. No run failed or timed out.

The headline statistic is the median of the five process medians. Detailed
quartiles, extrema, individual process medians, and all raw samples are in the
generated `summary.csv` and `raw_samples.csv` artifacts.

Across the 17 complete-output-correct kernels, raised CPU is a median 363.03x
slower than optimized vanilla C and raised GPU is a median 14.46x slower.

Exact upstream native-MFEM CUDA counterparts were subsequently captured for
all 17 kernels with the same shape, inputs, initial outputs, warmup count,
fresh-process count, sample count, clock state, and synchronized wall-clock
call boundary. All 85 native processes completed and retained 1,700 samples.
The native timing-run checksum agrees with vanilla C for every kernel under
the validation tolerance. Raised GPU is a median 85.37x slower than native
MFEM CUDA across the 17 kernels (range 8.78x to 214.40x).

These results expose the cost of many small pairwise library calls, runtime
planning, host-mapped buffers, residual host computation, and intermediate
materialization. Native MFEM uses its resident, specialized CUDA kernels while
the raised path is an unfused sequence of library calls.

## Protocol

- Jetson AGX Orin #2, driver 615.06, MAXN.
- CPU locked at 2.2016 GHz, GPU at 1.3005 GHz, EMC at 3.199 GHz.
- FP64, `NE=1024`, `D1D=4`, `Q1D=5`.
- Deterministic nonconstant inputs and nonzero initial outputs.
- One executable at a time; no parallel benchmark execution.
- Five independent processes per implementation.
- Five warmups followed by twenty retained wall-clock samples per process.
- GPU samples end after `cudaDeviceSynchronize`.
- Implementation order rotates between process sets.
- cuTensorNet network composition disabled because Mass3D composition failed
  complete-output validation; individual pairwise matches pass.
- `integrate_value_3d` excluded because its NE=1024 raised pipeline is
  incorrect before matching and its first match triggers an incompatible
  submap cast.
- Native MFEM sources are upstream low-level PA/DFEM CUDA kernels at MFEM
  revision `951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5`, cross-compiled for
  AArch64 `sm_87` with CUDA 12.6.85.
- MFEM's low-level DFEM headers are hidden by an MPI integration guard in this
  non-MPI build. The native harness supplies only the source-equivalent
  dispatch/data definitions needed to instantiate the unchanged upstream
  `interpolate.hpp` and `integrate.hpp` kernel bodies.

The 17 paths account for 126 validated static contraction sites. Complete
elementwise validation is recorded in `../FULL_OUTPUT_VALIDATION_2026-09-08.md`.
Timing-run final checksums are stable across processes and differ from vanilla
C only by expected floating-point reassociation error (largest raised-path
difference `1.615e-13`; largest native-MFEM difference `3.516e-13`).

The native-vs-vanilla audit here is a complete-output checksum check, not an
independent elementwise native-output comparison. The normalized/raised paths
retain their separate complete elementwise validation. Basis, quadrature-axis,
and gradient-component layouts are explicitly converted at the native harness
boundary so both implementations receive the same logical nonconstant inputs.

## Provenance limitation

This campaign used the audited shared working tree at base commit
`5464d8898667dbb9278f461336cc7ecec28f94ef`, which was dirty. Relevant tool and
binary hashes are retained, but this is not a clean committed-source campaign.
Do not call it final publication evidence until the intended changes are
committed and the campaign is reproduced from that pinned revision.

## Retained artifacts

- concise comparison: `../section42_campaign/performance_20260908.csv`
- generated detailed summary: external campaign artifact `summary.csv`
- generated 5,100-row sample ledger: external campaign artifact
  `raw_samples.csv`
- 255 immutable per-run logs, binary hashes, board metadata, progress log, and
  campaign driver output retained with the external campaign artifacts
- native summary, merged four-column comparison, checksum audit, 1,700-row
  sample ledger, 85 per-run logs, binary hashes, and board metadata retained in
  `/home/arjaiswal/mfem-ne1024-benchmark/results/native-clean-20260908/`
- reproducible native build/run/summarization drivers:
  `scripts/correctness/build_mfem_native_cuda_campaign.sh`,
  `scripts/correctness/run_mfem_native_cuda_campaign.sh`, and
  `scripts/correctness/summarize_mfem_native_cuda_campaign.py`
