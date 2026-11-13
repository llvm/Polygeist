/* batchnorm_batched.c — batched, per-channel batch normalization (inference).
 *
 * Extracted form of darknet's forward_batchnorm_layer (inference mode).
 * Same lift-friendly conventions as conv2d_batched.c / maxpool_batched.c:
 * scalar-int loop bounds via polybench-style dataset macros, perfect
 * nested affine for-loops, no scalar accumulator inside the body.
 *
 * The inference-mode formula collapses normalize + scale + bias into a
 * single fused element-wise op (cuDNN's cudnnBatchNormalizationForwardInference
 * does exactly this — the running stats are pre-computed, so there is no
 * cross-element reduction):
 *
 *   out[b,c,h,w] = scale[c] * (in[b,c,h,w] - mean[c]) * inv_std[c] + bias[c]
 *
 * where inv_std[c] = 1.0 / sqrt(var[c] + eps) is precomputed by the caller.
 *
 * Shape: NCHW. Iters: 4-parallel (B, C, H, W). Zero reductions.
 *
 * For a real ResNet conv2_x batchnorm: B=32, C=64, H=W=56.
 */
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
# define DATA_TYPE float
#endif

#if defined(MINI_DATASET)
# define B    4
# define C    8
# define H    32
# define W    32
#elif defined(LARGE_DATASET)
# define B    32
# define C    64
# define H    56
# define W    56
#else
# define B    4
# define C    8
# define H    32
# define W    32
#endif

/* The kernel. 4-deep parallel nest. Each output element reads:
 *   - in[b,c,h,w]
 *   - scale[c], mean[c], inv_std[c], bias[c]   (per-channel params)
 * and writes one out element. No reductions, so raise produces a single
 * linalg.generic with iter_types=[par×4] and 5 inputs.
 */
void kernel_batchnorm_batched(DATA_TYPE A[B][C][H][W],
                               DATA_TYPE scale[C],
                               DATA_TYPE mean[C],
                               DATA_TYPE inv_std[C],
                               DATA_TYPE bias[C],
                               DATA_TYPE Bout[B][C][H][W]) {
  int b, c, h, w;

  #pragma scop
  for (b = 0; b < B; ++b)
    for (c = 0; c < C; ++c)
      for (h = 0; h < H; ++h)
        for (w = 0; w < W; ++w)
          Bout[b][c][h][w] =
              scale[c] * (A[b][c][h][w] - mean[c]) * inv_std[c] + bias[c];
  #pragma endscop
}
