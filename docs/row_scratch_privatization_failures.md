# PrivatizeRowScratchAllocaForLoop — Failure Catalogue

The pattern is *implemented* in `lib/polygeist/Passes/RaiseToLinalg.cpp`
but is **NOT** currently registered in the raise pipeline — the
registration line is commented out, with a comment pointing at this
file. This document records what happens when the pattern *is* enabled,
so a future implementer knows exactly which kernels regress and why.

To re-enable for experimentation, uncomment the relevant line in
`runOnOperation` (search for `PrivatizeRowScratchAllocaForLoop`).

Date: 2026-05-16. Sweeps: PolyBench (30 kernels), MachSuite (19),
NPB-polybenchified (7). All other test inputs (BLAS, stress) unchanged.

## Net result: 4 regressions, 0 improvements

| kernel                | baseline           | with pattern        |
|-----------------------|--------------------|---------------------|
| **mg-psinv** (NPB ex) | PARTIAL_LIFT 3LG/2AF | **RAISE_FAIL (timeout)** |
| **mg-resid** (NPB ex) | PARTIAL_LIFT 3LG/2AF | **RAISE_FAIL (timeout)** |
| **mg-rprj3** (NPB ex) | PARTIAL_LIFT 3LG/2AF | **RAISE_FAIL (timeout)** |
| **fft-transpose** (MachSuite) | PARTIAL_LIFT 2LG/11AF | **RAISE_FAIL (timeout)** |

Every other kernel (29 PolyBench + 18 other MachSuite + 4 other NPB
extracted) is bit-identical to baseline. The pattern did not improve any
kernel; it strictly regressed 4.

## Failure mode (uniform across the 4 regressions)

1. cgeist emits the kernel as expected.
2. The raise-to-linalg pipeline starts.
3. `PrivatizeRowScratchAllocaForLoop` fires successfully on an outer
   `affine.for` containing a rank-1 static `memref.alloca`, rewriting
   the alloca to `memref<? x N x T>` and adding a per-iteration
   `memref.subview ... -> memref<N x T, strided<[1], offset:?>>`.
4. Greedy driver continues: `DistributeAffineForOnLinalgGeneric` and
   `AffineForOpRaising` each fire once or twice on the new IR.
5. `AffineForOpRaising` starts processing a deeper loop nest, begins
   emitting `affine.apply` + `polygeist.submap` ops, and never finishes.
6. Polygeist-opt is killed by the sweep's 60-second timeout.

`--debug-only=greedy-rewriter` traces confirm: total of 7 successful
pattern applications, then a long tail of failed-match attempts on
unchanged ops. Not a true infinite re-fire loop; the inner pattern's
polyhedral analysis is *very* slow on the post-privatization IR shape.

## Root-cause hypothesis (best guess; not fully verified)

The post-privatization rowView is

```mlir
%row = memref.subview %new[%iv, 0] [1, %N] [1, 1]
     : memref<? x N x T> to memref<N x T, strided<[1], offset: ?>>
```

The dynamic `offset: ?` in the strided layout type appears to defeat
`AffineForOpRaising`'s dep-check. The existing rank-0
`PrivatizeScratchAllocaForLoop` instead uses `polygeist.submap` to
express row-selection — and that path doesn't trigger the same
slowdown. So the next attempt should rewrite users via
`polygeist.submap` (passing `%iv` as an extra symbol) rather than
`memref.subview`.

## Failure-by-failure detail

### NPB-polybenchified/mg-psinv

Baseline raised IR (working without pattern):

```mlir
%alloca   = memref.alloca() : memref<35xf64>
%alloca_0 = memref.alloca() : memref<35xf64>
affine.for %i3 = 1 to N-1 {
  affine.for %i2 = 1 to N-1 {
    linalg.generic outs(%alloca_0 : memref<35xf64>) ...    // pass 1 fill (a)
    linalg.generic outs(%alloca   : memref<35xf64>) ...    // pass 1 fill (b)
    linalg.generic ins(... subviews of alloca/alloca_0 ...)
                   outs(... subview of arg1 ...)            // pass 2
  }
}
```

After pattern fires (with all patterns enabled), polygeist-opt times out
inside `AffineForOpRaising` on the inner i1 loop. The pattern's rewrite
is structurally fine — verified by running with `DistributeAffineForOnLinalgGeneric`
*disabled*, which produces clean post-rewrite IR (mg_psinv goes to
1LG/3AF residual). With Distribute enabled, the pipeline hangs.

### NPB-polybenchified/mg-resid

Identical shape to mg-psinv. Same failure mode.

### NPB-polybenchified/mg-rprj3

Identical shape (restriction operator with row scratch).
Same failure mode.

### MachSuite/fft-transpose

```mlir
%alloca   = memref.alloca() : memref<576xf64>
%alloca_5 = memref.alloca() : memref<8xf64>
%alloca_6 = memref.alloca() : memref<8xf64>
%alloca_7 = memref.alloca() : memref<512xf64>
%alloca_8 = memref.alloca() : memref<512xf64>
%alloca_9 = memref.alloca() : memref<8xi32>
```

Multiple rank-1 static scratch allocas. Pattern fires on at least one.
Then polygeist-opt is killed by the 60-second sweep timeout. Note this
is a regression on a benchmark where the C source has *much* less
clean a structure than mg_psinv — it's the bit-reversal FFT with lots
of imperative control flow — yet the pattern still fires because it
only requires "static rank-1 alloca, first touch is a write". The
match is too eager.

## What the pattern correctly *doesn't* affect

PolyBench (all 30 kernels) and the remaining MachSuite + NPB-extracted
kernels show *no* status change between baseline and pattern-enabled.
That means the recogniser is at least conservative enough to not
trigger on most code. The 4 regressions are specifically kernels with
the right structural shape.

## Tests confirming no improvements

- PolyBench gramschmidt: 5LG/1AF PARTIAL in both. (Has a column-vector
  scratch; the pattern doesn't recognize the access shape — uses
  `affine.load`/`store` directly into the multi-dim array, not a 1-D
  alloca that's separately allocated.)
- PolyBench durbin: 3LG/1AF PARTIAL in both. (Uses scalar carries
  (`alpha`/`beta`) — should be handled by the existing rank-0
  pattern; my new rank-1 pattern is irrelevant.)
- PolyBench correlation/covariance: unchanged.

So even on the PolyBench kernels we hoped to fix (durbin, gramschmidt),
the pattern doesn't fire because they don't have rank-1 *separately
allocated* scratch arrays. They use direct indexing into the original
matrix.

## Required follow-ups (in priority order)

1. **Re-emit users via `polygeist.submap` instead of `memref.subview`.**
   Mirror the 0-D pattern's rewrite. Should fix the AffineForOpRaising
   slowdown.
2. **Tighten match conditions.** The MachSuite/fft-transpose regression
   shows the recognizer fires on inputs that aren't the intended pattern.
   Add a precondition that the alloca is used in *at least two* sibling
   inner loops (the "fill then consume" shape) — that rules out
   single-loop scratch reads which don't benefit from privatization.
3. **Cover the PolyBench scratch patterns.** durbin and gramschmidt
   use direct multi-dim indexing rather than a separate scratch
   alloca — the pattern shape there is "use an outer loop's iv to
   index into the original 2-D array". Different transformation
   needed (not array privatization — closer to loop interchange or
   scalar promotion).

## Status

Pattern is implemented in `RaiseToLinalg.cpp` (~250 LOC) but registration
is commented out so the raise pipeline is bit-identical to baseline.
The 4 regressions above only manifest when the registration is
uncommented. This was the deliberate trade-off agreed with the user:
keep the work as a scaffold for a future fix, don't ship a strict
regression in the pipeline today.
