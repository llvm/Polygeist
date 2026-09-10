# PVA semantic fixture corpus

This directory contains independently authored plain-C descriptions of 31 PVA
Solutions image-operation families, three YOLOv5 stages beyond ordinary resize,
and ten common PVA-DL operation families.  The fixtures do not include or copy
gated NVIDIA source, do not call PVA APIs, and are not vendor conformance tests.

The operation name selects a function for the test harness only.  Compiler
recognition must be invariant under renaming and must derive exclusively from
types, operations, indexing, iterator/reduction structure, constants, control
flow, and data dependencies.  A fixture is not an automatic match merely
because cgeist translates it or the raising pipeline creates Linalg.

Some SDK family names cover many algorithms and parameters.  This corpus makes
the tested subset explicit: 3x3 dilation for morphology, SSD template matching,
one weighted-centroid CornerSubPix step, a 4x4 block-linear layout for BlToPl,
Manhattan distance transform, nearest-neighbor perspective warp, Sobel plus
dual threshold without Canny hysteresis, iterative label propagation for CCL,
and boundary extraction rather than ordered contour-chain construction.  These
subsets require comparison with the vendor API contract before they can count
as exact PVA matches.

Run the translation/raising/debufferization inventory with:

```sh
bash scripts/correctness/audit_pva_semantic_fixtures.sh /tmp/pva_audit
```

The audit output is evidence about compiler representability, not replacement
coverage.  Pattern matching and PVA lowering require a real external symbol,
complete ABI legality checks, positive correctness tests, rename-invariance
tests, and structurally similar negative tests.
