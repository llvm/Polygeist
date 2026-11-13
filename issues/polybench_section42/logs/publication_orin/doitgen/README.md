# Publication Orin campaign: doitgen

Raised GPU correctness and one-process 5+5 timing are complete. Native GPU is
unavailable because the retained adapter defines two project-authored CUDA
computational kernels and is ineligible for paper data.

- Raised correctness: PASS for all 3,360,000 canonical LARGE/FP64 values,
  `max_abs=0.01`, `max_rel=0.000337495781`.
- Compute-device median: 822.029358 ms (818.393494–827.211548,
  IQR 4.585266).
- Synchronized compute-wall median: 822.019584 ms.
- Memory-device median: 12.219136 ms.
- End-to-end median: 834.724256 ms (831.213504–839.952352,
  IQR 4.729984).

All compilation and linking occurred on x86. The timed executable combines the
fresh ABI-only 5+5 harness with the already retained correctness-approved
compiler-generated, coalesced/resident GPU object. Object and executable hashes
are recorded in `build-provenance.sha256`. No project-authored computational
kernel is used. No native speedup is reported.

The fresh generic residency rebuild attempt is retained in
`raised-timing-build.log`; it stopped at unsupported tensor `affine.for`
bufferization. This did not invalidate the retained compiler-generated GPU
object used here. Fixed hardware-state reporting was unavailable, so timing is
retained but publication-pending.
