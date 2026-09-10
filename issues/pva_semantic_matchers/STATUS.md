# PVA semantic fixture and matcher status

Date: 2026-09-10

Branch/worktree: `pva-structural-matchers` at
`/home/arjaiswal/Polygeist-pva-matchers`.

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
U8, and the U8 histogram are now selected as complete replacements. All three
pass the full path:
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

Box, bilateral, and histogram equalization remain gated out. Their causes are
non-equivalent vendor fixed coefficients, cross-shaped support plus quantized
approximate bilateral math, and Q15 equalization scaling. A valid vendor call
is therefore not sufficient evidence of semantic replacement.

The adapter and lowering layer supports 18 vendor ABI combinations and the
direct vendor-contract harness ran all 18 on Orin: box, Gaussian and morphology
for U8/S8/U16/S16; bilateral for U8; histogram for U8/U16 inputs and U32/S32
outputs; and histogram equalization for U8. Seventeen are exact against an
independently implemented vendor-contract reference. Bilateral is tolerance
only (123 pixels differ by one), so it is not an integer-exact pass. See
`silicon_dtype_matrix.csv` and `automatic_pipeline_silicon.csv`.

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
  families (`GaussianFilter3x3U8`, `MorphologyDilate3x3U8`,
  `ImageHistogramU8/U32`);
- structurally identified but production-rejected after silicon equivalence
  failure: 3 operation families;
- typed adapter/lowering ABI combinations built and exercised directly: 18;
- direct combinations exact against the vendor contract: 17;
- tolerance-only direct combinations: 1 (bilateral U8).

Extending the exact families to every vendor dtype in automatic source
matching requires source fixtures (or real programs) that preserve signedness
semantics for those dtypes. Box, bilateral and equalization must not be enabled
for the current fixtures unless a proof of the vendor's exact fixed-point
contract is present in the raised IR.

Raw normalized inventories are retained in `raising_summary.csv` and
`matcher_summary.csv`; the scripts regenerate both analyses.
