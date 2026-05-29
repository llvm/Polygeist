/* darknet_im2col_gemm.c — extracted Darknet convolution in its original
 * im2col + GEMM decomposition.
 *
 * Unlike third_party/darknet/src/convolutional_layer.c, this file keeps the
 * im2col helper and the GEMM helper in the same translation unit as the
 * kernel. That lets cgeist's inliner expose the full producer/consumer pair:
 *
 *   guarded im2col(data_im -> workspace) followed by GEMM(workspace -> out)
 *
 * The point is not to beat the direct-convolution extracted kernel; it is a
 * small same-TU fixture for developing the GuardedIm2Col + GEMM -> Conv2D
 * matcher.
 */

#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#if defined(MINI_DATASET)
#define IC  3
#define OC  4
#define H   8
#define W   8
#define KS  3
#elif defined(LARGE_DATASET)
#define IC  16
#define OC  16
#define H   32
#define W   32
#define KS  3
#else
#define IC  3
#define OC  4
#define H   8
#define W   8
#define KS  3
#endif

#define STRIDE 1
#define PAD    1
#define OH     ((H + 2 * PAD - KS) / STRIDE + 1)
#define OW     ((W + 2 * PAD - KS) / STRIDE + 1)
#define KCOL   (IC * KS * KS)
#define NCOL   (OH * OW)

static DATA_TYPE im2col_get_pixel(DATA_TYPE *im, int height, int width,
                                  int row, int col, int channel, int pad) {
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width)
    return (DATA_TYPE)0;
  return im[col + width * (row + height * channel)];
}

static void im2col_cpu(DATA_TYPE *data_im, int channels, int height, int width,
                       int ksize, int stride, int pad, DATA_TYPE *data_col) {
  int c, h, w;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;

  for (c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (h = 0; h < height_col; ++h) {
      for (w = 0; w < width_col; ++w) {
        int im_row = h_offset + h * stride;
        int im_col = w_offset + w * stride;
        int col_index = (c * height_col + h) * width_col + w;
        data_col[col_index] = im2col_get_pixel(
            data_im, height, width, im_row, im_col, c_im, pad);
      }
    }
  }
}

static void gemm_nn(int M, int N, int K, DATA_TYPE alpha, DATA_TYPE *A,
                    int lda, DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      DATA_TYPE a_part = alpha * A[i * lda + k];
      for (j = 0; j < N; ++j)
        C[i * ldc + j] += a_part * B[k * ldb + j];
    }
  }
}

void kernel_darknet_im2col_gemm(int channels, int height, int width,
                                int out_channels, int ksize, int stride,
                                int pad, DATA_TYPE input[IC * H * W],
                                DATA_TYPE weights[OC * KCOL],
                                DATA_TYPE workspace[KCOL * NCOL],
                                DATA_TYPE output[OC * NCOL]) {
  int i;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int ncol = height_col * width_col;
  int kcol = channels * ksize * ksize;

#pragma scop
  for (i = 0; i < out_channels * ncol; ++i)
    output[i] = (DATA_TYPE)0;

  im2col_cpu(input, channels, height, width, ksize, stride, pad, workspace);

  gemm_nn(out_channels, ncol, kcol, (DATA_TYPE)1, weights, kcol, workspace,
          ncol, output, ncol);
#pragma endscop
}

static void init_array(DATA_TYPE input[IC * H * W],
                       DATA_TYPE weights[OC * KCOL]) {
  int c, h, w, oc, kh, kw;
  for (c = 0; c < IC; ++c)
    for (h = 0; h < H; ++h)
      for (w = 0; w < W; ++w)
        input[w + W * (h + H * c)] =
            (DATA_TYPE)((c * 13 + h * 7 + w) % 19) / (DATA_TYPE)19;

  for (oc = 0; oc < OC; ++oc)
    for (c = 0; c < IC; ++c)
      for (kh = 0; kh < KS; ++kh)
        for (kw = 0; kw < KS; ++kw)
          weights[kw + KS * (kh + KS * (c + IC * oc))] =
              (DATA_TYPE)((oc * 5 + c * 3 + kh * 2 + kw) % 17) /
              (DATA_TYPE)17;
}

static void print_array(DATA_TYPE output[OC * NCOL]) {
  int oc, oh, ow;
  for (oc = 0; oc < OC; ++oc)
    for (oh = 0; oh < OH; ++oh)
      for (ow = 0; ow < OW; ++ow)
        fprintf(stderr, "%0.4f\n", output[ow + OW * (oh + OH * oc)]);
}

#ifdef MAIN
int main(void) {
  DATA_TYPE *input = malloc(sizeof(DATA_TYPE) * IC * H * W);
  DATA_TYPE *weights = malloc(sizeof(DATA_TYPE) * OC * KCOL);
  DATA_TYPE *workspace = malloc(sizeof(DATA_TYPE) * KCOL * NCOL);
  DATA_TYPE *output = malloc(sizeof(DATA_TYPE) * OC * NCOL);

  init_array(input, weights);
  kernel_darknet_im2col_gemm(IC, H, W, OC, KS, STRIDE, PAD, input, weights,
                             workspace, output);
  print_array(output);

  free(input);
  free(weights);
  free(workspace);
  free(output);
  return 0;
}
#endif
