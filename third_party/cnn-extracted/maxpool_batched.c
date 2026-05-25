/* maxpool_batched.c — batched, multi-channel 2D max pooling (forward).
 *
 * Extracted form of darknet's forward_maxpool_layer body. Same lift-
 * friendly conventions as conv2d_batched.c: scalar-int loop bounds via
 * polybench-style dataset macros.
 *
 * Layout: NCHW. Stride S, window K. Output H' = (H - K) / S + 1.
 *
 * For a real ResNet stem maxpool: B=32, C=64, H=W=112, K=3, S=2 → 56×56.
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
# define KS   2
# define STR  2
#elif defined(LARGE_DATASET)
# define B    32
# define C    64
# define H    112
# define W    112
# define KS   3
# define STR  2
#else
# define B    4
# define C    8
# define H    32
# define W    32
# define KS   2
# define STR  2
#endif

#define OH ((H - KS) / STR + 1)
#define OW ((W - KS) / STR + 1)

#define NEG_INF (-3.4028234e38f)

/* The kernel. 6-deep loop nest. Same two-pass pattern as conv2d_batched:
 *   - init: out[b,c,oh,ow] = -INF
 *   - reduce: out[b,c,oh,ow] = max(out, A[b,c,oh*S+kh,ow*S+kw])
 *
 * The init produces a 4-parallel linalg.generic. The reduce produces a
 * 4-parallel + 2-reduction linalg.generic with body `max(Out, In(0))`.
 */
void kernel_maxpool_batched(DATA_TYPE A[B][C][H][W],
                             DATA_TYPE Bout[B][C][OH][OW]) {
  int b, c, oh, ow, kh, kw;

  #pragma scop
  /* Init to -infinity */
  for (b = 0; b < B; ++b)
    for (c = 0; c < C; ++c)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          Bout[b][c][oh][ow] = NEG_INF;

  /* Max-reduce over the K×K window. Use the ternary form (lowers to
   * arith.select) instead of an if/then store — the if branch makes
   * cgeist emit a conditional store inside the inner loop, which the
   * raise pass leaves as affine.for. The ternary keeps the loop body
   * pure-arith so the whole 6-deep nest folds into one linalg.generic.
   */
  for (b = 0; b < B; ++b)
    for (c = 0; c < C; ++c)
      for (oh = 0; oh < OH; ++oh)
        for (ow = 0; ow < OW; ++ow)
          for (kh = 0; kh < KS; ++kh)
            for (kw = 0; kw < KS; ++kw) {
              DATA_TYPE v = A[b][c][oh * STR + kh][ow * STR + kw];
              DATA_TYPE cur = Bout[b][c][oh][ow];
              Bout[b][c][oh][ow] = (v > cur) ? v : cur;
            }
  #pragma endscop
}
