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

Run the complete corpus with:

```sh
ATEN_C_SWEEP_OUT=/tmp/aten_c_kernel_raise \
  bash scripts/correctness/aten_c_kernel_sweep.sh
```
