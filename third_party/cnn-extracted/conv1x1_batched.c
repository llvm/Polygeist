/* conv1x1_batched.c — batched 1×1 convolution. Mathematically a
 * per-pixel matmul: (B·H·W, IC) × (IC, OC) → (B·H·W, OC).
 *
 * cuDNN's K=1 conv path is generic (no Winograd, no IMPLICIT_PRECOMP_GEMM
 * specialisation); the matcher's lowering detects K=1 statically from
 * the filter's last two dims and routes to cublasDgemm instead, which
 * gets tensor cores on Ampere+.
 *
 * NCHW, FP32, no padding, stride 1, K=1.
 */
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
# define DATA_TYPE float
#endif

#if defined(MINI_DATASET)
# define B  4
# define IC 16
# define OC 16
# define H  32
# define W  32
#elif defined(LARGE_DATASET)
# define B  32
# define IC 256
# define OC 256
# define H  56
# define W  56
#else
# define B  4
# define IC 16
# define OC 16
# define H  32
# define W  32
#endif
#define KS 1
#define OH H
#define OW W

void kernel_conv1x1_batched(DATA_TYPE A[B][IC][H][W],
                              DATA_TYPE F[OC][IC][KS][KS],
                              DATA_TYPE Bout[B][OC][OH][OW]) {
  int b, oc, ic, oh, ow;

  #pragma scop
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][oc][oh][ow] = 0;

  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          for (ic = 0; ic < IC; ++ic)
            Bout[b][oc][oh][ow] += A[b][ic][oh][ow] * F[oc][ic][0][0];
  #pragma endscop
}
