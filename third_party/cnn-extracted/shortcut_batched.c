/* shortcut_batched.c — batched residual-add shortcut layer.
 *
 * Extracted form of darknet's forward_shortcut_layer (matched-shape case).
 * ResNet's identity shortcut: out = out + src, where both tensors share
 * the same NCHW shape. Same lift-friendly conventions as the other
 * cnn-extracted files.
 *
 * Body: out[b,c,h,w] = src[b,c,h,w] + out[b,c,h,w]. 4-parallel iter
 * domain (B, C, H, W), zero reductions. cuDNN side this maps to a
 * cudnnAddTensor call, or with the existing matcher library it lines up
 * with a generic elementwise add.
 *
 * Default MINI shape matches the other extracted kernels (B=4, C=8,
 * H=W=32). LARGE = ResNet conv2_x output (B=32, C=64, H=W=56).
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

/* The kernel. 4-deep parallel nest. Each output element reads one src
 * value and one current-out value, writes one out value. */
void kernel_shortcut_batched(DATA_TYPE A[B][C][H][W],
                              DATA_TYPE Bout[B][C][H][W]) {
  int b, c, h, w;

  #pragma scop
  for (b = 0; b < B; ++b)
    for (c = 0; c < C; ++c)
      for (h = 0; h < H; ++h)
        for (w = 0; w < W; ++w)
          Bout[b][c][h][w] = A[b][c][h][w] + Bout[b][c][h][w];
  #pragma endscop
}
