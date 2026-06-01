/* stencil_conv2d_3x3.c -- image/PDE-style 2D stencil fixtures.
 *
 * These kernels are intentionally written as straight-line 3x3 neighbourhood
 * expressions so the raise pipeline can expose them as one linalg.generic with
 * nine shifted input subviews. The matcher should lower those to the generic
 * @cudnnConvolution2D_9tap library entry with the coefficients surfaced as
 * scalar launch operands.
 */

#include <stdio.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef STENCIL_H
#define STENCIL_H 64
#endif

#ifndef STENCIL_W
#define STENCIL_W 64
#endif

#ifndef REPEAT
#define REPEAT 50
#endif

#ifndef STENCIL_KERNEL
#define STENCIL_KERNEL kernel_stencil_box3x3
#endif

void kernel_stencil_box3x3(int h, int w,
                           DATA_TYPE in[STENCIL_H][STENCIL_W],
                           DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)0.11111111 * in[i - 1][j - 1] +
          (DATA_TYPE)0.11111111 * in[i - 1][j] +
          (DATA_TYPE)0.11111111 * in[i - 1][j + 1] +
          (DATA_TYPE)0.11111111 * in[i][j - 1] +
          (DATA_TYPE)0.11111111 * in[i][j] +
          (DATA_TYPE)0.11111111 * in[i][j + 1] +
          (DATA_TYPE)0.11111111 * in[i + 1][j - 1] +
          (DATA_TYPE)0.11111111 * in[i + 1][j] +
          (DATA_TYPE)0.11111111 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_gaussian3x3(int h, int w,
                                DATA_TYPE in[STENCIL_H][STENCIL_W],
                                DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)0.0625 * in[i - 1][j - 1] +
          (DATA_TYPE)0.1250 * in[i - 1][j] +
          (DATA_TYPE)0.0625 * in[i - 1][j + 1] +
          (DATA_TYPE)0.1250 * in[i][j - 1] +
          (DATA_TYPE)0.2500 * in[i][j] +
          (DATA_TYPE)0.1250 * in[i][j + 1] +
          (DATA_TYPE)0.0625 * in[i + 1][j - 1] +
          (DATA_TYPE)0.1250 * in[i + 1][j] +
          (DATA_TYPE)0.0625 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_sobel_x3x3(int h, int w,
                               DATA_TYPE in[STENCIL_H][STENCIL_W],
                               DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)-1.0 * in[i - 1][j - 1] +
          (DATA_TYPE)0.0 * in[i - 1][j] +
          (DATA_TYPE)1.0 * in[i - 1][j + 1] +
          (DATA_TYPE)-2.0 * in[i][j - 1] +
          (DATA_TYPE)0.0 * in[i][j] +
          (DATA_TYPE)2.0 * in[i][j + 1] +
          (DATA_TYPE)-1.0 * in[i + 1][j - 1] +
          (DATA_TYPE)0.0 * in[i + 1][j] +
          (DATA_TYPE)1.0 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_sobel_y3x3(int h, int w,
                               DATA_TYPE in[STENCIL_H][STENCIL_W],
                               DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)-1.0 * in[i - 1][j - 1] +
          (DATA_TYPE)-2.0 * in[i - 1][j] +
          (DATA_TYPE)-1.0 * in[i - 1][j + 1] +
          (DATA_TYPE)0.0 * in[i][j - 1] +
          (DATA_TYPE)0.0 * in[i][j] +
          (DATA_TYPE)0.0 * in[i][j + 1] +
          (DATA_TYPE)1.0 * in[i + 1][j - 1] +
          (DATA_TYPE)2.0 * in[i + 1][j] +
          (DATA_TYPE)1.0 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_laplacian4_3x3(int h, int w,
                                   DATA_TYPE in[STENCIL_H][STENCIL_W],
                                   DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)0.0 * in[i - 1][j - 1] +
          (DATA_TYPE)1.0 * in[i - 1][j] +
          (DATA_TYPE)0.0 * in[i - 1][j + 1] +
          (DATA_TYPE)1.0 * in[i][j - 1] +
          (DATA_TYPE)-4.0 * in[i][j] +
          (DATA_TYPE)1.0 * in[i][j + 1] +
          (DATA_TYPE)0.0 * in[i + 1][j - 1] +
          (DATA_TYPE)1.0 * in[i + 1][j] +
          (DATA_TYPE)0.0 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_laplacian8_3x3(int h, int w,
                                   DATA_TYPE in[STENCIL_H][STENCIL_W],
                                   DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)1.0 * in[i - 1][j - 1] +
          (DATA_TYPE)1.0 * in[i - 1][j] +
          (DATA_TYPE)1.0 * in[i - 1][j + 1] +
          (DATA_TYPE)1.0 * in[i][j - 1] +
          (DATA_TYPE)-8.0 * in[i][j] +
          (DATA_TYPE)1.0 * in[i][j + 1] +
          (DATA_TYPE)1.0 * in[i + 1][j - 1] +
          (DATA_TYPE)1.0 * in[i + 1][j] +
          (DATA_TYPE)1.0 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_sharpen3x3(int h, int w,
                               DATA_TYPE in[STENCIL_H][STENCIL_W],
                               DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)0.0 * in[i - 1][j - 1] +
          (DATA_TYPE)-1.0 * in[i - 1][j] +
          (DATA_TYPE)0.0 * in[i - 1][j + 1] +
          (DATA_TYPE)-1.0 * in[i][j - 1] +
          (DATA_TYPE)5.0 * in[i][j] +
          (DATA_TYPE)-1.0 * in[i][j + 1] +
          (DATA_TYPE)0.0 * in[i + 1][j - 1] +
          (DATA_TYPE)-1.0 * in[i + 1][j] +
          (DATA_TYPE)0.0 * in[i + 1][j + 1];
#pragma endscop
}

void kernel_stencil_emboss3x3(int h, int w,
                              DATA_TYPE in[STENCIL_H][STENCIL_W],
                              DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 1; i < h - 1; ++i)
    for (j = 1; j < w - 1; ++j)
      out[i][j] =
          (DATA_TYPE)-2.0 * in[i - 1][j - 1] +
          (DATA_TYPE)-1.0 * in[i - 1][j] +
          (DATA_TYPE)0.0 * in[i - 1][j + 1] +
          (DATA_TYPE)-1.0 * in[i][j - 1] +
          (DATA_TYPE)1.0 * in[i][j] +
          (DATA_TYPE)1.0 * in[i][j + 1] +
          (DATA_TYPE)0.0 * in[i + 1][j - 1] +
          (DATA_TYPE)1.0 * in[i + 1][j] +
          (DATA_TYPE)2.0 * in[i + 1][j + 1];
#pragma endscop
}

/* Negative-control fixture for the next matcher extension: cuDNN can run a
 * 5x5 convolution, but the current matcher only has the 3x3/9-tap template.
 */
void kernel_stencil_box5x5(int h, int w,
                           DATA_TYPE in[STENCIL_H][STENCIL_W],
                           DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          (DATA_TYPE)0.04 * in[i - 2][j - 2] +
          (DATA_TYPE)0.04 * in[i - 2][j - 1] +
          (DATA_TYPE)0.04 * in[i - 2][j] +
          (DATA_TYPE)0.04 * in[i - 2][j + 1] +
          (DATA_TYPE)0.04 * in[i - 2][j + 2] +
          (DATA_TYPE)0.04 * in[i - 1][j - 2] +
          (DATA_TYPE)0.04 * in[i - 1][j - 1] +
          (DATA_TYPE)0.04 * in[i - 1][j] +
          (DATA_TYPE)0.04 * in[i - 1][j + 1] +
          (DATA_TYPE)0.04 * in[i - 1][j + 2] +
          (DATA_TYPE)0.04 * in[i][j - 2] +
          (DATA_TYPE)0.04 * in[i][j - 1] +
          (DATA_TYPE)0.04 * in[i][j] +
          (DATA_TYPE)0.04 * in[i][j + 1] +
          (DATA_TYPE)0.04 * in[i][j + 2] +
          (DATA_TYPE)0.04 * in[i + 1][j - 2] +
          (DATA_TYPE)0.04 * in[i + 1][j - 1] +
          (DATA_TYPE)0.04 * in[i + 1][j] +
          (DATA_TYPE)0.04 * in[i + 1][j + 1] +
          (DATA_TYPE)0.04 * in[i + 1][j + 2] +
          (DATA_TYPE)0.04 * in[i + 2][j - 2] +
          (DATA_TYPE)0.04 * in[i + 2][j - 1] +
          (DATA_TYPE)0.04 * in[i + 2][j] +
          (DATA_TYPE)0.04 * in[i + 2][j + 1] +
          (DATA_TYPE)0.04 * in[i + 2][j + 2];
#pragma endscop
}

static DATA_TYPE input_img[STENCIL_H][STENCIL_W];
static DATA_TYPE output_img[STENCIL_H][STENCIL_W];

static DATA_TYPE init_value(int i, int j) {
  int v = (i * 17 + j * 13 + 7) % 101;
  return (DATA_TYPE)((v - 50) * 0.01f);
}

static void init_arrays(void) {
  for (int i = 0; i < STENCIL_H; ++i) {
    for (int j = 0; j < STENCIL_W; ++j) {
      input_img[i][j] = init_value(i, j);
      output_img[i][j] = (DATA_TYPE)0;
    }
  }
}

static void print_checksum(void) {
  DATA_TYPE checksum = (DATA_TYPE)0;
  for (int i = 0; i < STENCIL_H; ++i) {
    for (int j = 0; j < STENCIL_W; ++j) {
      checksum += output_img[i][j];
    }
  }
  printf("%.8f\n", (double)checksum);
}

int main(void) {
  init_arrays();
  for (int r = 0; r < REPEAT; ++r) {
    STENCIL_KERNEL(STENCIL_H, STENCIL_W, input_img, output_img);
  }
  print_checksum();
  return 0;
}
