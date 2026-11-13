# Reusing validated reference imports

For development batches, after sourcing `envsetup.sh`, run:

```
POLYGEIST_SPEC_SUITE=lib_v3 /usr/bin/python3 scripts/correctness/lib_v2/build_lib_v2.py --incremental -j 8
```

The first run raises and validates the references normally. Later runs reuse a
record only if Affine, memref-Linalg, and tensor validation previously passed,
all recorded generated files are unchanged, and the input fingerprint agrees.
The fingerprint covers the reference/manifest/upstream source, preprocessed
headers, recipe, import and validation helpers, compiler and validation tools,
link/runtime dependencies, and relevant environment. Missing, corrupt, failed,
or incomplete records receive fresh work. Changes during the import abort
publication of its catalog rather than recording mixed revisions.

The complete library and provenance manifest are emitted even when some entries
are reused. Reused entries carry `validation_reuse.input_sha256`; the summary
prints reused/fresh counts and elapsed time. The local cache is under
`build/reference-import-cache/` and is not an artifact dependency.

Omit `--incremental` for a full rebuild. In particular, **compilation-cost
measurements must use full builds**, not cached validation times. `--no-raise`
and `--no-validate` cannot be combined with `--incremental`. `--only` retains its
existing diagnostic behavior and does not publish the full library.

Batch new references before importing. Build their applications in parallel
against one stable compiler/catalog/runtime snapshot, then serialize GPU
measurements. Keep per-case correctness and source-adaptation checks. Run shared
regressions once per relevant shared implementation change, rather than after
each manifest-only addition. Existing failed references are retried, so this
cache does not eliminate the cost of unsupported portions of the catalog.

Regression tests: `scripts/correctness/tests/test_import_cache.py`.
Real-source and complete-catalog comparisons: `issues/import_reuse_20261111/`.
