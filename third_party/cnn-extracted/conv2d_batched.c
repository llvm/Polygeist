/* conv2d_batched.c — batched, multi-channel 2D convolution (forward).
 *
 * The polybenchGpu conv2d is single-batch, single-channel, fixed 3×3 — the
 * worst possible shape for cuDNN. This file extracts a "real" CNN conv
 * layer: batch + channels + filter loop. ResNet-style. Polybench-style
 * harness so cgeist can lift it via affine.for.
 *
 * Direct convolution form (no im2col). The 7-deep loop nest below is what
 * cuDNN's IMPLICIT_PRECOMP_GEMM algorithm computes — just with cuBLAS
 * tiling instead of a naive loop. Matcher should recognise it as a
 * 4-parallel + 3-reduction tensor contraction (eventually mapping to
 * cublasDgemm via im2col, or directly to cudnnConvolutionForward).
 *
 * No padding, stride 1, no dilation, no activation. NCHW layout.
 *
 * Default MINI shape: B=4, C=8, H=W=32, K=3 (output H=W=30).
 *   Total flops:  4 × 8 × 30² × 8 × 9 = 207360
 *   Total input data: 4 × 8 × 32² × 4 = 128 KB
 *
 * LARGE shape (ResNet-50 conv2 size): B=32, C=64, H=W=56, K=3 (output 54²).
 *   Total flops:  32 × 64 × 54² × 64 × 9 ≈ 3.4 GFLOPs
 *   Total data ≈ 30 MB
 */

#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
# define DATA_TYPE float
#endif

/* Polybench-style dataset macros. Pick one via -D{MINI,LARGE,XLARGE}_DATASET */
#if defined(MINI_DATASET)
# define B    4
# define IC   8
# define OC   8
# define H    32
# define W    32
# define KS   3
#elif defined(LARGE_DATASET)
# define B    32
# define IC   64
# define OC   64
# define H    56
# define W    56
# define KS   3
#elif defined(XLARGE_DATASET)
# define B    32
# define IC   128
# define OC   128
# define H    28
# define W    28
# define KS   3
#else
/* default = MINI */
# define B    4
# define IC   8
# define OC   8
# define H    32
# define W    32
# define KS   3
#endif

#define OH (H - KS + 1)
#define OW (W - KS + 1)

/* Init inputs with a simple linear pattern so the output values are
 * predictable + check-summable. */
static void init_array(DATA_TYPE A[B][IC][H][W],
                       DATA_TYPE F[OC][IC][KS][KS]) {
  int b, c, i, j;
  for (b = 0; b < B; ++b)
    for (c = 0; c < IC; ++c)
      for (i = 0; i < H; ++i)
        for (j = 0; j < W; ++j)
          A[b][c][i][j] = (DATA_TYPE)((b + c + i + j) % 17) / (DATA_TYPE)17;
  for (b = 0; b < OC; ++b)
    for (c = 0; c < IC; ++c)
      for (i = 0; i < KS; ++i)
        for (j = 0; j < KS; ++j)
          F[b][c][i][j] = (DATA_TYPE)((b * 3 + c * 5 + i * 7 + j) % 11)
                          / (DATA_TYPE)11;
}

static void print_array(DATA_TYPE Bout[B][OC][OH][OW]) {
  int b, c, i, j;
  for (b = 0; b < B; ++b)
    for (c = 0; c < OC; ++c)
      for (i = 0; i < OH; ++i) {
        for (j = 0; j < OW; ++j)
          fprintf(stderr, "%0.4f ", Bout[b][c][i][j]);
        if ((b * OC * OH + c * OH + i) % 20 == 0) fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}

/* The kernel. 7-deep loop nest:
 *   for each (batch, out_channel, oh, ow) — parallel
 *     for each (in_channel, kh, kw) — reduction
 *       acc += A[b][ic][oh+kh][ow+kw] * F[oc][ic][kh][kw]
 *
 * Loop bounds are all macros expanded to compile-time constants, so cgeist
 * lifts to affine.for cleanly (no struct-field-load issue).
 */
void kernel_conv2d_batched(DATA_TYPE A[B][IC][H][W],
                            DATA_TYPE F[OC][IC][KS][KS],
                            DATA_TYPE Bout[B][OC][OH][OW]) {
  int b, oc, ic, oh, ow, kh, kw;

  /* Two-pass form: explicit init nest (4 parallel) followed by the
   * accumulation nest (4 parallel + 3 reduction). The init makes the
   * accumulation form a perfect 7-deep nest with no scalar temp — the
   * raise-affine-to-linalg pass needs this to fold all four outer
   * parallel loops into the linalg.generic instead of leaving them as
   * imperative affine.for with iter_args.
   */
  #pragma scop
  /* Init: Bout = 0 */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][oc][oh][ow] = 0;

  /* Accumulate */
  for (b = 0; b < B; ++b)
    for (oc = 0; oc < OC; ++oc)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          for (ic = 0; ic < IC; ++ic)
            for (kh = 0; kh < KS; ++kh)
              for (kw = 0; kw < KS; ++kw)
                Bout[b][oc][oh][ow] +=
                    A[b][ic][oh + kh][ow + kw] * F[oc][ic][kh][kw];
  #pragma endscop
}

#ifdef MAIN
int main(void) {
  DATA_TYPE (*A)[IC][H][W] = malloc(sizeof(DATA_TYPE) * B * IC * H * W);
  DATA_TYPE (*F)[IC][KS][KS] = malloc(sizeof(DATA_TYPE) * OC * IC * KS * KS);
  DATA_TYPE (*Bout)[OC][OH][OW] = malloc(sizeof(DATA_TYPE) * B * OC * OH * OW);

  init_array(A, F);
  kernel_conv2d_batched(A, F, Bout);
  print_array(Bout);

  free(A); free(F); free(Bout);
  return 0;
}
#endif
