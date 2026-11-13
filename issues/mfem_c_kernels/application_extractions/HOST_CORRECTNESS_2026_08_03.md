# MFEM larger-application correctness debug — updated 2026-08-04

All harness-supported executable MFEM application extractions now pass the
complete raise, debufferize, matcher, ABI-lowering, LLVM, wrapper, and runtime
pipeline on both the host CPU shim and the attached Jetson.

Common specialization: FP64, `NE=2`, `D1D=4`, `Q1D=5`.  These are deliberately
small correctness fixtures, not performance-sized GPU workloads.

| Function | Library launches | Host gate | Warm Jetson gate |
|---|---:|---|---|
| `mfem_app_mtop_iso_elasticity_dfem_2d` | 12 | PASS, exact | PASS, `3.47e-18` |
| `mfem_app_dfem_minimal_surface_2d` | 6 | PASS, exact | PASS, `8.67e-19` |
| `mfem_app_ex35p_h1_3d` | 19 | PASS, `6.94e-18` | PASS, `6.94e-18` |
| `mfem_app_ex35p_hcurl_3d` | 29 | PASS, `4.86e-17` | PASS, `4.86e-17` |
| `mfem_app_ex35p_hdiv_3d` | 12 | PASS, `4.86e-17` | PASS, `4.86e-17` |
| `mfem_app_ex9p_mass_convection_2d` | 9 | PASS, `6.94e-18` | PASS, `6.94e-18` |
| `mfem_app_grad_div_3d` | 12 | PASS, `4.86e-17` | PASS, `4.86e-17` |
| `mfem_app_abs_l1_mass_3d` | 5 | PASS, `6.94e-18` | PASS, `6.94e-18` |
| `mfem_app_abs_l1_diffusion_3d` | 14 | PASS, exact | PASS, `2.71e-20` |
| `mfem_app_abs_l1_curlcurl_3d` | 29 | PASS, `4.86e-17` | PASS, `4.86e-17` |

`mfem_app_navier_tgv_pa_operators_3d` is now also executable.  Scratch-sliced
normalizations for vector mass, vector diffusion, discrete gradient, and
nonlinear vector convection reduce its residual loops from 26 to zero.  The
raised IR contains 720 Linalg ops, the matcher emits 70 cuTensorNet calls, and
the independent direct-C host comparison passes with
`max_abs=max_rel=4.163336e-17`.

## Failure causes and fixes

The original failures were independent bugs that happened to surface in the
same end-to-end table:

- **Partial submap treated as a reshape.** A contiguous view was incorrectly
  assumed to cover its full base tensor.  This produced invalid narrowing
  casts and wrong second-half write-back.  Fast reshape lowering now requires
  statically proven full coverage; partial views use affine materialization.
- **Dropped multi-output results.** Recursive debufferization retained only
  one result from some H(curl)/H(div) multi-output generics.  The application
  pipeline now uses joint multi-root debufferization.
- **Incomplete contraction ABI metadata.** Output slices lost physical
  strides and affine constant base offsets.  The lowering now preserves both,
  and snapshots opaque-call results so live results do not alias one buffer.
- **Unsafe computed-submap library calls.** One-shot bufferization can choose
  an earlier aliased base for an opaque raw-pointer call.  Until library calls
  are represented by a bufferizable op, contractions over computed submap
  bases remain residual Linalg instead of being emitted unsafely.
- **Malformed DAXPBY match.** Rank-N/two-operand pointwise candidates were
  emitted against a rank-1/four-operand ABI.  The matcher now emits only the
  exact supported form and supplies its scalar coefficients.
- **Store-to-load dependence lost during raising.** ex9 stored `new_r` and
  reloaded `r[i]` for a dot product; raising made the reload an independent
  old-value input.  Exact-address post-store loads are now forwarded from the
  stored SSA value.
- **Jetson host-registration cache exhaustion.** Larger graphs exceeded the
  fixed 256-entry persistent registration cache.  A full cache now evicts and
  unregisters the least-recently used completed-call mapping.

The corrected warm silicon measurements are recorded in
`../silicon_results/2026-08-03_mfem_applications_jetson.log` and published in
the MFEM CE page.  Because these fixtures are tiny and each recognized stage
becomes a separate cuTensorNet plan/launch, the timings are dominated by host
planning and launch overhead; they are correctness evidence, not speedup
evidence.
