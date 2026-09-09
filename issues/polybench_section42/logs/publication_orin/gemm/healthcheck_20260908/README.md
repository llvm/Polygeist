# Raised GEMM Orin health check (2026-09-08)

The unchanged correctness-approved AArch64/sm_87 FP64 direct-DGEMM executable (`sha256 52e8ca37be4e0be1c0ddcc016a0e50b4e9af69a0bef424d555a5014df892dc2f`) was deployed after an idle-device preflight and rerun on Orin.

Result: `PASS values=1100000 failures=0 max_abs=0.01 max_rel=4.46388715e-05` with the established `rtol=5e-4`, `atol=1.1e-2` gate.

The run reported 103.852989 ms compute-device and 194.993440 ms harness E2E.  This is a single correctness health check, not a five-warmup/five-sample publication measurement, so these observed times do not replace the accepted ledger.

- `gemm-healthcheck.silicon.log.gz`: complete raw Orin output.
- `gemm-healthcheck.runner.log`: transport/preflight and execution log.
- `gemm-healthcheck.compare.log`: complete-output comparison result.
