/* conv_bn_relu_batched.c — fused-pattern test kernel.
 *
 * Chains the three operations that make up the inner of a ResNet
 * residual block (conv → bn → relu) into a single C function. Polybench-
 * style. Goal: matcher should fold all four loop nests (init + conv +
 * bn + relu) into one fused launch — `cudnnConvolutionBiasActivation
 * Forward`-shaped — so the bandwidth-bound bn + relu ride the compute-
 * bound conv's GPU win instead of paying their own per-call setup.
 *
 * NCHW, FP32, no padding, stride 1, K×K filter. OH = H - K + 1,
 * OW = W - K + 1. BN is the inference-mode formula with pre-baked
 * inv_std = 1/sqrt(var+eps). ReLU uses the ternary form so it lowers
 * to arith.select (the if-form would leave residual affine.for).
 */
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
# define DATA_TYPE float
#endif

#if defined(MINI_DATASET)
# define B  4
# define IC 8
# define OC 8
# define H  32
# define W  32
# define KS 3
#elif defined(LARGE_DATASET)
# define B  32
# define IC 64
# define OC 64
# define H  56
# define W  56
# define KS 3
#else
# define B  4
# define IC 8
# define OC 8
# define H  32
# define W  32
# define KS 3
#endif
#define OH (H - KS + 1)
#define OW (W - KS + 1)

/* Four-loop-nest body. Each nest is a separate linalg.generic after
 * raising. The matcher's job is to fold all four into one launch. */
void kernel_conv_bn_relu_batched(
    DATA_TYPE A[B][IC][H][W],
    DATA_TYPE F[OC][IC][KS][KS],
    DATA_TYPE scale[OC],
    DATA_TYPE mean[OC],
    DATA_TYPE inv_std[OC],
    DATA_TYPE bias[OC],
    DATA_TYPE Bout[B][OC][OH][OW]) {
  int b, oc, ic, oh, ow, kh, kw;

  #pragma scop
  /* (1) Init: Bout = 0 */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][oc][oh][ow] = 0;

  /* (2) Conv: Bout += A * F */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          for (ic = 0; ic < IC; ++ic)
            for (kh = 0; kh < KS; ++kh)
              for (kw = 0; kw < KS; ++kw)
                Bout[b][oc][oh][ow] +=
                    A[b][ic][oh + kh][ow + kw] * F[oc][ic][kh][kw];

  /* (3) BN (in-place): Bout = scale*(Bout - mean)*inv_std + bias */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][oc][oh][ow] =
              scale[oc] * (Bout[b][oc][oh][ow] - mean[oc]) * inv_std[oc]
              + bias[oc];

  /* (4) ReLU (in-place ternary): Bout = max(Bout, 0) */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow) {
          DATA_TYPE v = Bout[b][oc][oh][ow];
          Bout[b][oc][oh][ow] = (v > 0.0f) ? v : 0.0f;
        }
  #pragma endscop
}
