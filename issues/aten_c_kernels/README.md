# ATen C kernel extraction corpus

These are standalone C reference forms of high-value ATen operations. They
deliberately expose the numerical loop body using fixed, small shapes so the
experiment measures raising and GPU-library matching rather than PyTorch's C++
dispatch, `TensorIterator`, registration, or object lifetime machinery.

The corpus is pinned to PyTorch revision
`d7af122d81a49b1fa7a31ba52bd57c026f092646`. Each source names the corresponding
upstream `aten/src/ATen/native` implementation family. These files preserve the
numerical algorithm but replace ATen tensor/dispatch machinery with explicit C
arrays and dimensions; they are compiler fixtures, not replacements for ATen's
public ABI.

The original ten-kernel batch covers established library routes:

| File | ATen operation | Intended GPU route |
| --- | --- | --- |
| `aten_mm.c` | `aten::mm` | cuBLAS GEMM |
| `aten_addmm.c` | `aten::addmm` | cuBLAS GEMM with alpha/beta |
| `aten_mv.c` | `aten::mv` | cuBLAS GEMV |
| `aten_dot.c` | `aten::dot` | cuBLAS dot |
| `aten_add.c` | in-place `aten::add` | cuDNN add-tensor/custom CUDA |
| `aten_softmax.c` | `aten::_softmax` | cuDNN softmax |
| `aten_rms_norm.c` | `aten::rms_norm` | custom CUDA RMSNorm |
| `aten_conv2d.c` | `aten::conv2d` | cuDNN convolution |
| `aten_max_pool2d.c` | `aten::max_pool2d` | cuDNN pooling |
| `aten_batch_norm.c` | inference `aten::batch_norm` | cuDNN batch normalization |

The second batch adds fifteen structurally distinct operations:

- linear algebra: `bmm`, `outer`
- reductions: `sum`, `mean`
- activations: `relu`, tanh-approximation `gelu`, `silu`
- normalization/loss: `layer_norm`, mean-reduced `mse_loss`
- pooling: `avg_pool2d`, specialized `adaptive_avg_pool2d`
- data movement: materialized transpose, `im2col`, nearest-neighbor upsample
- vector algebra: batched three-component `cross`

The third batch adds twenty-five more native numerical families:

- activations: sigmoid, tanh, leaky-ReLU, ELU, softplus, hard-sigmoid,
  hard-swish, hard-tanh, and clamp
- elementwise/loss/reduction: lerp, L1 loss, binary cross entropy, product,
  and cumulative sum
- indexing/layout: embedding, channel shuffle, pixel shuffle, reflection
  padding, and replication padding
- pooling/convolution/resampling: 3D average and adaptive-average pooling,
  1D and 3D convolution, 2D transposed convolution, and bilinear upsampling

This is a 50-family numerical extraction corpus, not a claim that ATen has only
50 operations. The exhaustive direct translation-unit sweep is archived in
`notes/polygeist_raise_to_linalg/aten_raise_sweep_2026_07_21`: it covers all
224 portable `aten/src/ATen/native` C/C++ translation units selected at the
pinned revision. Only 8 emitted a target function and none raised to Linalg,
primarily because cgeist cannot yet lower ATen's C++ dispatch, temporary, and
cleanup machinery. The standalone files isolate the numerical algorithms
behind that frontend boundary.

Run the complete corpus with:

```sh
bash scripts/correctness/aten_c_kernel_sweep.sh
```

The default output is the checked-in `results/` directory. Each kernel keeps
`orig.mlir`, `raised.mlir`, `debuf.mlir`, `matched.mlir`, and diagnostics; flat
MLIR aliases are also generated for the IR explorer.
