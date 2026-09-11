/* conv_bias_relu_add_batched.c — fused conv + bias + residual + relu.
 *
 * Canonical ResNet output stage. The matcher should fold all five loop
 * nests (init + conv + bias + residual-add + relu) into one launch and
 * route to cudnnConvolutionBiasActivationForward — whose API natively
 * supports y = activation(α₁·conv(x,w) + α₂·z + bias).
 *
 * NCHW, FP32, no padding, stride 1, K×K filter.
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

void kernel_conv_bias_relu_add_batched(
    DATA_TYPE A[B][IC][H][W],
    DATA_TYPE F[OC][IC][KS][KS],
    DATA_TYPE bias[OC],
    DATA_TYPE Z[B][OC][OH][OW],
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

  /* (3) Bias (per-output-channel, broadcast over B/OH/OW) */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][oc][oh][ow] += bias[oc];

  /* (4) Residual-add: Bout += Z (skip connection) */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][oc][oh][ow] += Z[b][oc][oh][ow];

  /* (5) ReLU (ternary form) */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow) {
          DATA_TYPE v = Bout[b][oc][oh][ow];
          Bout[b][oc][oh][ow] = (v > 0.0f) ? v : 0.0f;
        }
  #pragma endscop
}
