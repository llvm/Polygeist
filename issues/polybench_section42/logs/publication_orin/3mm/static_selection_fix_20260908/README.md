# Generic static-function selection fix: 3mm

The ordinary fresh-source build command was used without a user-supplied `-Dstatic=` flag.  The build driver detected that explicit cgeist selection had silently omitted source-local `kernel_3mm`, retained the empty affine artifact, and automatically retried with translation-only static-linkage exposure.

Results:

- six matched launches: three external-library zero fills and three GEMMs;
- 18 lowered runtime calls including pipeline boundaries;
- AArch64/sm_87 executable cross-built entirely on the x86 host;
- Orin correctness: `PASS values=880000 failures=0 max_abs=0 max_rel=0`.

`build.log`, the complete compressed silicon output, comparison, hashes, and each relevant IR stage are retained here.  The observed single-run E2E value is diagnostic and is not a publication 5+5 timing.
