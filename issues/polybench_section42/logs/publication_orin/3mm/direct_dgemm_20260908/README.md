# 3mm direct-DGEMM rerun (2026-09-08)

Status: **fresh source-selection failure diagnosed; exposed fresh kernel and stored-IR paths pass; no new performance result accepted**.

The initial fresh LARGE/FP64 source-to-matcher cross-build found zero ABI-lowerable launches, retained residual Linalg, and failed to link because `kernel_3mm_impl` was absent.  The cause is now known: canonical `kernel_3mm` is declared `static`, and current explicit `cgeist --function=kernel_3mm` silently emits an empty module.  Consequently, the 79-byte deployment error and one-token comparison produced after that failed build are invalid downstream artifacts, retained only as failure evidence.

To isolate the direct-DGEMM runtime change, the previously retained semantic matched IR was cross-built with the current runtime.  Its first run emitted three real direct `cublasDgemm` calls but returned 880,000 wrong values.  After an independent exact-shape cuBLAS probe exercised the device stack, the exact same executable was rerun without rebuilding and passed all 880,000 outputs exactly.  The first result is now classified as a transient cold/stale CUDA-state failure rather than a deterministic direct-DGEMM or compiler failure.

As a fresh-source diagnostic, compiling with `-Dstatic=` exposes the selected kernel without changing its computation.  The current pipeline then finds six launches (three fills and three GEMMs), emits 18 runtime calls, cross-builds, and passes all 880,000 outputs exactly.  The intended generic fix is a missing-symbol fallback to wildcard cgeist emission followed by the existing structural `--select-func`, not a permanent benchmark-specific macro.

Evidence:

- `raised-correctness-build.log`: fresh matcher and link failure.
- `raised-correctness.silicon.log` and `raised-correctness.compare.log`: invalid downstream artifacts from the missing executable.
- `stored-raised-correctness-build.log`: stored semantic-IR control cross-build.
- `stored-raised-correctness.silicon.log.gz` and `stored-raised-correctness.compare.log`: valid Orin correctness failure.
- `ir/`: fresh no-match and stored matched control IR.
- `build-provenance.sha256` and `stored-build-provenance.sha256`: retained hashes.
- `debug_matrix/`: passing stored/fresh diagnostic outputs and the fresh static-exposure build log.

Execution used the local `pva-general` transport profile.  Private route and credential material are intentionally not retained in the repository.
