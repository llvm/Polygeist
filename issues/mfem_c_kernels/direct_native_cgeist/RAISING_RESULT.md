# Raising the isolated native MFEM mass kernel

Date: 2026-09-09

## Outcome

The isolated CUDA source now passes cgeist translation and the existing MFEM
raising pipeline, but it does **not** yet expose complete tensor contractions
to the library matcher.

| Stage | Status | Result |
|---|---|---|
| Native extraction after flat-index normalization | PASS | AArch64/SM87 build and full 16,384-element Orin comparison pass with zero error |
| cgeist CUDA-to-structured translation | PASS | valid 282-line MLIR module; five floating-point multiply sites retained |
| SCF-to-affine conversion | PASS | 11 affine loops and 12 remaining SCF loops before the Linalg pipeline |
| affine-to-Linalg pipeline | PASS | one `linalg.generic`; 22 affine and 14 SCF/parallel/while loops remain |
| multi-root debufferization | PASS | output verifies with `polygeist-opt` |
| semantic matcher | PASS, no match | `0 matched / 1 bodies` |
| structured-region analysis | REJECTED | one region rejected; no proved producer/consumer fusion |

## What was raised

The one `linalg.generic` is only the final length-five reduction:

```text
dd += QD[qy][dx] * Bt[dy][qy]
Y[dx,dy,e] += dd
```

It is not represented as a self-contained two-input contraction: the raised
body still captures loads from shared scratch memrefs instead of exposing them
as Linalg inputs. Consequently the matcher correctly reports no cuTENSOR or
cuTensorNet candidate.

## What was not raised

The complete MFEM operation has four mathematical contraction stages:

```text
DQ(dy,qx) = sum_dx X(dy,dx) * B(qx,dx)
QQ(qy,qx) = D(qx,qy,e) * sum_dy DQ(dy,qx) * B(qy,dy)
QD(qy,dx) = sum_qx QQ(qy,qx) * Bt(dx,qx)
Y(dx,dy,e) += sum_qy QD(qy,dx) * Bt(dy,qy)
```

Only the scalar reduction in the fourth stage becomes Linalg. The first three
and their producer/consumer composition remain explicit loops.

## Root cause

cgeist faithfully retains the CUDA execution mechanics. The
`MFEM_FOREACH_THREAD` loops become thread-strided SCF loops nested under the
block/thread parallel structure, and barriers split the shared-memory stages.
The reused `sm0`/`sm1` buffers also carry cross-stage aliasing. The current
affine raiser cannot turn that mixed parallel/SCF/shared-memory representation
into complete Linalg operands and indexing maps.

Trying to raise the raw GPU form directly is worse: `--raise-scf-to-affine`
asserts in `AffineApplyNormalizer` because a GPU thread-derived value is not a
valid affine symbol. CPUifying with `distribute` avoids the assertion and
produces valid MLIR, but retains the one-trip thread-stride loops described
above.

This is a raising limitation, not evidence that the contractions are absent.
No compiler implementation was modified for this audit. A legitimate fix
should structurally collapse provably one-trip CUDA thread loops, model barrier
separated shared-memory stages, and preserve their dependencies before running
affine-to-Linalg.

Run `run_isolated_raise.sh` to reproduce the pipeline and counts.
