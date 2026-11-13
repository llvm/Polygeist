# INVALID: complete-build timing summary

These measurements used the wrong generic submap/debufferization profile and
must not be published.  See `INVALID_COMPLETE_BUILD_CAMPAIGNS_20260911.md`.

This summary uses only the three Egglog-enabled complete-build attempts per
input.  Each successful attempt compiled a fresh C input through cgeist,
affine-to-Linalg raising, debufferization, matching, library-definition and ABI
lowering, MLIR/LLVM lowering, AArch64 object generation, runtime/harness
compilation, and final linking.  Temporary intermediate artifacts were deleted
after every attempt.  The runtime shim was recompiled in every attempt.

For each input, the reported value is the median of its three attempts.  The
suite mean, median, minimum, and maximum summarize those per-input medians.
Only inputs that linked successfully in all three attempts enter the timing
distribution; failed attempts and their logs remain in the raw campaigns.

| Benchmark | Timed / attempted inputs | Mean (s) | Median (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|---:|
| PolyBench | 28 / 30 | 9.012 | 8.960 | 8.550 | 9.460 |
| Llama | 17 / 23 | 8.510 | 8.520 | 7.860 | 9.130 |
| MFEM | 9 / 31 | 27.314 | 22.400 | 11.170 | 52.040 |

The corresponding attempt counts are 84 successful and 6 failed for
PolyBench, 51 successful and 18 failed for Llama, and 27 successful and 66
failed for MFEM.  There were no outer timeouts.

Median stage times, computed by first taking each input's three-run median and
then taking the suite median, are:

| Benchmark | Frontend | Raising | Matching process | Library lowering | Native/runtime compile | Link |
|---|---:|---:|---:|---:|---:|---:|
| PolyBench | 0.102 s | 0.024 s | 0.944 s | 0.176 s | 6.642 s | 0.706 s |
| Llama | 0.076 s | 0.018 s | 0.853 s | 0.167 s | 6.581 s | 0.400 s |
| MFEM | 3.975 s | 0.177 s | 3.910 s | 1.660 s | 10.439 s | 0.105 s |

The complete stage CSV also includes definition injection, downstream
lowering, and ABI-wrapper preparation.

Failure coverage:

- PolyBench: `adi`, `doitgen`.
- Llama: `extended_forward`, `gate_up_projection`, `kv_cache_rw`,
  `llama2c_rmsnorm`, `llama2c_softmax`, `qkv_projection`.
- MFEM: 22 of the 31 fixtures failed consistently; exact rows and diagnostics
  are retained in the MFEM campaign.

All campaigns used revision `7d69dcc9928448e775d9e671992aa92e25386754`
and recorded exact compiler/source hashes.  The x86 compilation host was
shared and loaded: start load averages were 31.08/19.77/17.22 for PolyBench,
13.14/13.27/14.35 for Llama, and 14.95/15.04/14.80 for MFEM on 24 logical
CPUs.  These must therefore be labelled loaded-host results; they are not a
clean idle-host absolute compile-time measurement.
