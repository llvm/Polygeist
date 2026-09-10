# PVA semantic fixture and matcher status

Setup, acquisition boundaries, and end-to-end reproduction instructions are
documented in [`README.md`](README.md).

Date: 2026-09-10

Development branch: `pva-structural-matchers`.

## Scope and provenance

The corpus contains 31 independently authored image-operation functions and
13 YOLO/PVA-DL functions.  No PVA Solutions implementation source or header was
copied.  The gated headers were inspected only to confirm public API symbol
availability.  Function names are harness entry points only and are never
inputs to recognition.

The image fixtures cover Morphology, ImageHistogram, CornerSubPix, MinMaxLoc,
TemplateMatching, MixChannels, BlToPl, ConvertImageFormat, ImageBlend,
ImageResize, BlurFilterROI, DistanceTransform, BilateralFilter,
BruteForceMatcher, WarpPerspective, ImageStats, CannyEdgeDetector,
MedianFilter, HistogramEqualization, BackgroundSubtractor, GaussianPyramid,
Remap, ORBDescExtractor, ImageFlip, CCL, HOG, FastCornerDetector,
ContourTracing, Conv2d, GaussianFilter, and BoxFilter.

The second source covers NHWC-to-NCHW letterbox normalization, YOLOv5 decode,
batched NMS, convolution, depthwise convolution, GEMM, max pooling, ReLU,
gather, quantize/dequantize, softmax, top-k, and layer normalization. Ordinary
ImageResize is already present in the image corpus.

## Compiler results

- C syntax/build check: 44/44 pass.
- cgeist translation: 44/44 pass.
- affine/Linalg raising pipeline: 44/44 pass.
- debufferization: 44/44 pass.
- functions with memref `linalg.generic`: 36/44.
- functions with tensor `linalg.generic`: 36/44.
- functions retaining at least one affine/SCF loop: 29/44.

The eight functions without tensor Linalg are CornerSubPix, BlurFilterROI,
DistanceTransform, WarpPerspective, MedianFilter, ORBDescExtractor,
FastCornerDetector, and fused NHWC-to-NCHW letterbox normalization.  They remain
valid translated loop IR, but require new raising representations.

## Bugs found and fixed in this worktree

1. cgeist lost unsignedness through typedefs such as `uint8_t` when converting
   integers to/from floating point.  It now canonicalizes the Clang type before
   selecting `arith.uitofp`/`arith.fptoui` versus signed operations.
2. A directly indexed `uint8_t` could become signed `arith.index_cast`.  Array
   subscripts now use `arith.index_castui` for unsigned source types.
3. RaiseToLinalg incorrectly parallelized cross-index same-buffer recurrences,
   including `cdf[i] = cdf[i-1] + histogram[i]`.  Such loops now remain
   sequential until a real scan representation exists.  This also removes the
   histogram-equalization debufferization nontermination observed before the
   fix.  Exact-address in-place updates and post-store reloads remain legal and
   retain the existing SSA forwarding behavior.

Focused regression tests cover unsigned typedef casts, unsigned array indices,
preservation of prefix-sum loops, and store-to-load forwarding.

The focused compiler tests pass. The six-candidate test includes renamed
functions, wrong-normalization and unrelated-operation negatives. Its
production half proves that only three integer-exact source replacements are
selected. The typed PVA lowering lit test covers all 18 supported ABI routes.

## Recognition results

The existing general matcher selects three replacements, but only two cover a
complete requested operation:

- ReLU -> `cutensorUnary_relu_f32` (complete external-library match).
- Softmax -> `cudnnSoftmaxForwardOut_tensor` (complete three-stage external
  library composition).
- HOG zero initialization -> `memset_zero_1D_f32` (partial subregion only; this
  is not a HOG match).

Other reported candidates are partial subregions only: histogram zeroing,
batched-NMS suppression initialization, and the inner dot product of GEMM. They
must not be counted as complete operation matches.

A read-only structural detector identifies six PVA operation candidates from
IR structure and constants:

- 3x3 U8 box filter -> `pvaBoxFilterCreate/Submit`;
- 3x3 U8 Gaussian filter -> `pvaGaussianFilterCreate/Submit`;
- 3x3 U8 bilateral filter -> `pvaBilateralFilterCreate/Submit`;
- 3x3 U8 morphological dilation -> `pvaMorphologyCreate/Submit`;
- 256-bin U8 histogram -> `pvaImageHistogramCreate/Submit`; and
- U8 histogram equalization -> `pvaHistogramEqualizationCreate/Submit`.

The production matcher uses no source/function names. It checks the complete
function ABI, loop/reduction structure, neighborhood, signedness operations,
constants and operation-specific dataflow. Gaussian U8, morphological dilation
U8/S8/U16/S16, and the U8 histogram are now selected as complete replacements.
All six dtype-specific replacements pass the full path:
C -> cgeist -> raised/debufferized MLIR -> structural matcher ->
`kernel.launch` -> typed PVA runtime call -> AArch64 executable -> Orin silicon,
with exact full-output comparison against the source semantics.

All six candidate paths were deliberately built and run once during the ABI
audit. This initially exposed four unsafe substitutions:

- box U8: 15,875 differing pixels, maximum error 1;
- Gaussian U8 with the former hardcoded sigma 1.0: 33,212 differing pixels,
  maximum error 5;
- bilateral U8: 18,245 differing pixels, maximum error 2; and
- histogram equalization U8: 23,675 differing pixels, maximum error 1.

The matcher now recovers the Gaussian coefficient axis, verifies symmetry and
separability, solves `sigma = sqrt(-1/(2*log(neighbor/center)))`, simulates the
PVA fixed-point coefficient construction, and emits sigma operands only if the
source and vendor integer weights and rounding are equivalent. For `[1,2,1]`,
it inferred sigma X/Y 0.8493218; the rebuilt automatic replacement passed Orin
with zero full-output differences. A perturbed-axis negative test is rejected.

Box, bilateral, and histogram equalization remain gated out by default. Their causes are
non-equivalent vendor fixed coefficients, cross-shaped support plus quantized
approximate bilateral math, and Q15 equalization scaling. A valid vendor call
is therefore not sufficient evidence of semantic replacement.

The matcher now represents this numerical boundary explicitly. Exact matches
carry `polygeist.numerical_contract = "exact"` and remain automatic. A
non-bit-exact match is emitted only when the user provides a per-operation
`--pva-approximation-budget OP=N`; the launch then carries
`polygeist.numerical_contract = "approximate"` and the requested maximum-error
budget. Enabling one operation cannot enable another. The generic build driver
accepts the comma-separated environment form
`POLYGEIST_PVA_APPROXIMATION_BUDGETS`.

All six opt-in datatype routes were rebuilt from their ordinary C fixtures
through the complete compiler pipeline and run on a Jetson AGX Orin. Box was within its
explicit budget for U8 (13,321 differing pixels, maximum error 1), S8 (22,943,
maximum 2), U16 (4,339, maximum 1), and S16 (24,031, maximum 2). Bilateral U8
was within budget 2 (18,245 differing pixels, maximum error 2), and histogram
equalization U8 was within budget 1 (23,675 differing pixels, maximum error 1).
Each comparison covered all 49,152 output pixels. These are empirical campaign
observations, not proofs that the same bounds hold for every input.
In particular, bilateral's source-comparison budget of 2 is distinct from the
PVA Solutions sample's tolerance of 1 against its own Q1.7 fixed-point CPU
reference.

The publication-oriented occurrence/type ledger is
`raised_c_numerical_contracts.csv`; the complete typed Box/Gaussian/histogram
build and silicon output is retained in `raised_all_dtypes_20260910.log`, with
the extracted summary in `raised_all_dtypes_20260910.csv`. The reproducible
driver is `scripts/correctness/run_pva_all_raised_dtypes.sh`. Bilateral and
equalization evidence remains in `raised_numerical_contracts_20260910.log`.

An impulse-response probe established the Box U8 discrepancy directly on
Orin. PVA uses the Q8 coefficient matrix `[29,28,29; 28,29,28; 29,28,28]`,
whose entries sum to 256. The source instead computes the symmetric operation
`(sum + 4) / 9`. For a value-255 impulse at the top-left, top-right,
bottom-left, or center tap, PVA returns 29 while the source returns 28. Thus
the maximum-one error is an inherent operator-semantic difference, not a
signedness, pointer, border, or harness bug. The raw observations and probe
source are `box_u8_coefficient_probe.csv` and `pva_box_coefficient_probe.c`.

The adapter and lowering layer supports 18 vendor ABI combinations and the
direct vendor-contract harness ran all 18 on Orin: box, Gaussian and morphology
for U8/S8/U16/S16; bilateral for U8; histogram for U8/U16 inputs and U32/S32
outputs; and histogram equalization for U8. Seventeen are exact against an
independently implemented vendor-contract reference. Bilateral is tolerance
only (123 pixels differ by one), so it is not an integer-exact pass. See
`silicon_dtype_matrix.csv` and `automatic_pipeline_silicon.csv`.

## Legacy ABI retirement

The old untyped image symbols `pva{Box,Gaussian,Bilateral}Filter_3x3_i8/i16`
and `pvaHistogramEqualization_i8` are now rejected atomically by
`--lower-kernel-launch-to-pva`. Their launch signatures lost signedness and,
for Gaussian and bilateral, did not carry sigma parameters. The associated
hardcoded PVA runtime entry points and approximate CPU substitutes were
removed. This also eliminates the unsupported signed/i16 bilateral route.

Only typed, parameterized PVA image symbols remain lowerable. Gaussian carries
the sigma values recovered and proven by structural matching. The production
matcher continues to withhold box, bilateral, and histogram equalization for
the current fixtures because their source arithmetic is not integer-exact with
the vendor contract. A regression test supplies all seven retired symbols and
requires an explicit pass failure for every one.

## Vendor availability boundary

The inspected PVA Solutions distribution contains Create/Submit APIs for 30 of
the 31 named image families.  No `ContourTracing` operator header or symbol was
found in that distribution, so the contour fixture currently has no confirmed
external-library route.  Several broad API labels have deliberately scoped
fixtures; their precise parameter/border/layout compatibility still needs
vendor-contract validation before an exact match claim.

YOLO decode and batched NMS are present as pipeline/sample concepts, not yet as
confirmed standalone PVA operator ABI routes.  The DL fixtures describe the
math embedded by DLInference, but replacing individual regions with
`pvaDLInference` requires a valid serialized PVA-DL program and cannot be
inferred from scalar math alone.

## Current executable boundary

- completed, automatically selected, source-exact PVA matches: 3 operation
  families / 12 dtype-specific routes: Gaussian and morphology at
  U8/S8/U16/S16, plus histogram at U8/U16 input and U32/S32 output;
- structurally identified but default-rejected after silicon equivalence
  failure: 3 operation families / 6 dtype-specific routes: Box at
  U8/S8/U16/S16, bilateral U8, and histogram equalization U8. Every route has
  a separately reported, explicitly enabled approximate execution;
- typed adapter/lowering ABI combinations built and exercised directly: 18;
- direct combinations exact against the vendor contract: 17;
- tolerance-only direct combinations: 1 (bilateral U8).

Every vendor-supported datatype route now has a raised source fixture and an
Orin execution. Gaussian, morphology, and histogram are exact across all their
supported datatype combinations. Because MLIR's signless `i32` cannot retain a
C histogram destination's U32/S32 distinction, that backend ABI is selected
explicitly with `--pva-histogram-output-type`; it is never inferred from a
function name. Box, bilateral, and equalization must not be enabled as exact
transformations for the current fixtures. Approximate execution requires an
explicit per-operation budget and is reported separately from exact coverage.

The morphology dtype campaign used nontrivial full-width inputs, including
negative values for S8/S16, and compared all 49,152 output elements. The Orin
results were exact for S8, U16 and S16 (`mismatches=0`, `max_error=0`), matching
the earlier U8 result. The reproducible cross-build/deployment driver is
`scripts/correctness/run_pva_morphology_raised_dtypes.sh`.

Raw normalized inventories are retained in `raising_summary.csv` and
`matcher_summary.csv`; the scripts regenerate both analyses.
