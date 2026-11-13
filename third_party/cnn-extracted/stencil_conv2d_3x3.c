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

/* 5x5 fixtures exercise the sibling 25-tap cuDNN convolution path. */
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

#define STENCIL5_TAP(DI, DJ, W) ((DATA_TYPE)(W) * in[i + (DI)][j + (DJ)])

void kernel_stencil_gaussian5x5(int h, int w,
                                DATA_TYPE in[STENCIL_H][STENCIL_W],
                                DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          STENCIL5_TAP(-2, -2, 0.00390625) +
          STENCIL5_TAP(-2, -1, 0.01562500) +
          STENCIL5_TAP(-2,  0, 0.02343750) +
          STENCIL5_TAP(-2,  1, 0.01562500) +
          STENCIL5_TAP(-2,  2, 0.00390625) +
          STENCIL5_TAP(-1, -2, 0.01562500) +
          STENCIL5_TAP(-1, -1, 0.06250000) +
          STENCIL5_TAP(-1,  0, 0.09375000) +
          STENCIL5_TAP(-1,  1, 0.06250000) +
          STENCIL5_TAP(-1,  2, 0.01562500) +
          STENCIL5_TAP( 0, -2, 0.02343750) +
          STENCIL5_TAP( 0, -1, 0.09375000) +
          STENCIL5_TAP( 0,  0, 0.14062500) +
          STENCIL5_TAP( 0,  1, 0.09375000) +
          STENCIL5_TAP( 0,  2, 0.02343750) +
          STENCIL5_TAP( 1, -2, 0.01562500) +
          STENCIL5_TAP( 1, -1, 0.06250000) +
          STENCIL5_TAP( 1,  0, 0.09375000) +
          STENCIL5_TAP( 1,  1, 0.06250000) +
          STENCIL5_TAP( 1,  2, 0.01562500) +
          STENCIL5_TAP( 2, -2, 0.00390625) +
          STENCIL5_TAP( 2, -1, 0.01562500) +
          STENCIL5_TAP( 2,  0, 0.02343750) +
          STENCIL5_TAP( 2,  1, 0.01562500) +
          STENCIL5_TAP( 2,  2, 0.00390625);
#pragma endscop
}

void kernel_stencil_sobel_x5x5(int h, int w,
                               DATA_TYPE in[STENCIL_H][STENCIL_W],
                               DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          STENCIL5_TAP(-2, -2, -5.0) +
          STENCIL5_TAP(-2, -1, -4.0) +
          STENCIL5_TAP(-2,  0,  0.0) +
          STENCIL5_TAP(-2,  1,  4.0) +
          STENCIL5_TAP(-2,  2,  5.0) +
          STENCIL5_TAP(-1, -2, -8.0) +
          STENCIL5_TAP(-1, -1, -10.0) +
          STENCIL5_TAP(-1,  0,  0.0) +
          STENCIL5_TAP(-1,  1,  10.0) +
          STENCIL5_TAP(-1,  2,  8.0) +
          STENCIL5_TAP( 0, -2, -10.0) +
          STENCIL5_TAP( 0, -1, -20.0) +
          STENCIL5_TAP( 0,  0,  0.0) +
          STENCIL5_TAP( 0,  1,  20.0) +
          STENCIL5_TAP( 0,  2,  10.0) +
          STENCIL5_TAP( 1, -2, -8.0) +
          STENCIL5_TAP( 1, -1, -10.0) +
          STENCIL5_TAP( 1,  0,  0.0) +
          STENCIL5_TAP( 1,  1,  10.0) +
          STENCIL5_TAP( 1,  2,  8.0) +
          STENCIL5_TAP( 2, -2, -5.0) +
          STENCIL5_TAP( 2, -1, -4.0) +
          STENCIL5_TAP( 2,  0,  0.0) +
          STENCIL5_TAP( 2,  1,  4.0) +
          STENCIL5_TAP( 2,  2,  5.0);
#pragma endscop
}

void kernel_stencil_sobel_y5x5(int h, int w,
                               DATA_TYPE in[STENCIL_H][STENCIL_W],
                               DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          STENCIL5_TAP(-2, -2, -5.0) +
          STENCIL5_TAP(-2, -1, -8.0) +
          STENCIL5_TAP(-2,  0, -10.0) +
          STENCIL5_TAP(-2,  1, -8.0) +
          STENCIL5_TAP(-2,  2, -5.0) +
          STENCIL5_TAP(-1, -2, -4.0) +
          STENCIL5_TAP(-1, -1, -10.0) +
          STENCIL5_TAP(-1,  0, -20.0) +
          STENCIL5_TAP(-1,  1, -10.0) +
          STENCIL5_TAP(-1,  2, -4.0) +
          STENCIL5_TAP( 0, -2,  0.0) +
          STENCIL5_TAP( 0, -1,  0.0) +
          STENCIL5_TAP( 0,  0,  0.0) +
          STENCIL5_TAP( 0,  1,  0.0) +
          STENCIL5_TAP( 0,  2,  0.0) +
          STENCIL5_TAP( 1, -2,  4.0) +
          STENCIL5_TAP( 1, -1,  10.0) +
          STENCIL5_TAP( 1,  0,  20.0) +
          STENCIL5_TAP( 1,  1,  10.0) +
          STENCIL5_TAP( 1,  2,  4.0) +
          STENCIL5_TAP( 2, -2,  5.0) +
          STENCIL5_TAP( 2, -1,  8.0) +
          STENCIL5_TAP( 2,  0,  10.0) +
          STENCIL5_TAP( 2,  1,  8.0) +
          STENCIL5_TAP( 2,  2,  5.0);
#pragma endscop
}

void kernel_stencil_laplacian5x5(int h, int w,
                                 DATA_TYPE in[STENCIL_H][STENCIL_W],
                                 DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          STENCIL5_TAP(-2, -2,  0.0) +
          STENCIL5_TAP(-2, -1,  0.0) +
          STENCIL5_TAP(-2,  0, -1.0) +
          STENCIL5_TAP(-2,  1,  0.0) +
          STENCIL5_TAP(-2,  2,  0.0) +
          STENCIL5_TAP(-1, -2,  0.0) +
          STENCIL5_TAP(-1, -1, -1.0) +
          STENCIL5_TAP(-1,  0, -2.0) +
          STENCIL5_TAP(-1,  1, -1.0) +
          STENCIL5_TAP(-1,  2,  0.0) +
          STENCIL5_TAP( 0, -2, -1.0) +
          STENCIL5_TAP( 0, -1, -2.0) +
          STENCIL5_TAP( 0,  0, 16.0) +
          STENCIL5_TAP( 0,  1, -2.0) +
          STENCIL5_TAP( 0,  2, -1.0) +
          STENCIL5_TAP( 1, -2,  0.0) +
          STENCIL5_TAP( 1, -1, -1.0) +
          STENCIL5_TAP( 1,  0, -2.0) +
          STENCIL5_TAP( 1,  1, -1.0) +
          STENCIL5_TAP( 1,  2,  0.0) +
          STENCIL5_TAP( 2, -2,  0.0) +
          STENCIL5_TAP( 2, -1,  0.0) +
          STENCIL5_TAP( 2,  0, -1.0) +
          STENCIL5_TAP( 2,  1,  0.0) +
          STENCIL5_TAP( 2,  2,  0.0);
#pragma endscop
}

void kernel_stencil_sharpen5x5(int h, int w,
                               DATA_TYPE in[STENCIL_H][STENCIL_W],
                               DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          STENCIL5_TAP(-2, -2, -0.125) +
          STENCIL5_TAP(-2, -1, -0.125) +
          STENCIL5_TAP(-2,  0, -0.125) +
          STENCIL5_TAP(-2,  1, -0.125) +
          STENCIL5_TAP(-2,  2, -0.125) +
          STENCIL5_TAP(-1, -2, -0.125) +
          STENCIL5_TAP(-1, -1,  0.250) +
          STENCIL5_TAP(-1,  0,  0.250) +
          STENCIL5_TAP(-1,  1,  0.250) +
          STENCIL5_TAP(-1,  2, -0.125) +
          STENCIL5_TAP( 0, -2, -0.125) +
          STENCIL5_TAP( 0, -1,  0.250) +
          STENCIL5_TAP( 0,  0,  1.000) +
          STENCIL5_TAP( 0,  1,  0.250) +
          STENCIL5_TAP( 0,  2, -0.125) +
          STENCIL5_TAP( 1, -2, -0.125) +
          STENCIL5_TAP( 1, -1,  0.250) +
          STENCIL5_TAP( 1,  0,  0.250) +
          STENCIL5_TAP( 1,  1,  0.250) +
          STENCIL5_TAP( 1,  2, -0.125) +
          STENCIL5_TAP( 2, -2, -0.125) +
          STENCIL5_TAP( 2, -1, -0.125) +
          STENCIL5_TAP( 2,  0, -0.125) +
          STENCIL5_TAP( 2,  1, -0.125) +
          STENCIL5_TAP( 2,  2, -0.125);
#pragma endscop
}

void kernel_stencil_emboss5x5(int h, int w,
                              DATA_TYPE in[STENCIL_H][STENCIL_W],
                              DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 2; i < h - 2; ++i)
    for (j = 2; j < w - 2; ++j)
      out[i][j] =
          STENCIL5_TAP(-2, -2, -2.0) +
          STENCIL5_TAP(-2, -1, -1.0) +
          STENCIL5_TAP(-2,  0, -1.0) +
          STENCIL5_TAP(-2,  1,  0.0) +
          STENCIL5_TAP(-2,  2,  0.0) +
          STENCIL5_TAP(-1, -2, -1.0) +
          STENCIL5_TAP(-1, -1, -1.0) +
          STENCIL5_TAP(-1,  0,  0.0) +
          STENCIL5_TAP(-1,  1,  1.0) +
          STENCIL5_TAP(-1,  2,  0.0) +
          STENCIL5_TAP( 0, -2, -1.0) +
          STENCIL5_TAP( 0, -1,  0.0) +
          STENCIL5_TAP( 0,  0,  1.0) +
          STENCIL5_TAP( 0,  1,  1.0) +
          STENCIL5_TAP( 0,  2,  1.0) +
          STENCIL5_TAP( 1, -2,  0.0) +
          STENCIL5_TAP( 1, -1,  1.0) +
          STENCIL5_TAP( 1,  0,  1.0) +
          STENCIL5_TAP( 1,  1,  1.0) +
          STENCIL5_TAP( 1,  2,  2.0) +
          STENCIL5_TAP( 2, -2,  0.0) +
          STENCIL5_TAP( 2, -1,  0.0) +
          STENCIL5_TAP( 2,  0,  1.0) +
          STENCIL5_TAP( 2,  1,  2.0) +
          STENCIL5_TAP( 2,  2,  2.0);
#pragma endscop
}

#undef STENCIL5_TAP

/* 7x7 fixture exercises the generalized packed-weight ntap path. */
#define STENCIL7_TAP(DI, DJ, W) ((DATA_TYPE)(W) * in[i + (DI)][j + (DJ)])

void kernel_stencil_box7x7(int h, int w,
                           DATA_TYPE in[STENCIL_H][STENCIL_W],
                           DATA_TYPE out[STENCIL_H][STENCIL_W]) {
  int i, j;
#pragma scop
  for (i = 3; i < h - 3; ++i)
    for (j = 3; j < w - 3; ++j)
      out[i][j] =
          STENCIL7_TAP(-3, -3, 0.02040816326530612) +
          STENCIL7_TAP(-3, -2, 0.02040816326530612) +
          STENCIL7_TAP(-3, -1, 0.02040816326530612) +
          STENCIL7_TAP(-3,  0, 0.02040816326530612) +
          STENCIL7_TAP(-3,  1, 0.02040816326530612) +
          STENCIL7_TAP(-3,  2, 0.02040816326530612) +
          STENCIL7_TAP(-3,  3, 0.02040816326530612) +
          STENCIL7_TAP(-2, -3, 0.02040816326530612) +
          STENCIL7_TAP(-2, -2, 0.02040816326530612) +
          STENCIL7_TAP(-2, -1, 0.02040816326530612) +
          STENCIL7_TAP(-2,  0, 0.02040816326530612) +
          STENCIL7_TAP(-2,  1, 0.02040816326530612) +
          STENCIL7_TAP(-2,  2, 0.02040816326530612) +
          STENCIL7_TAP(-2,  3, 0.02040816326530612) +
          STENCIL7_TAP(-1, -3, 0.02040816326530612) +
          STENCIL7_TAP(-1, -2, 0.02040816326530612) +
          STENCIL7_TAP(-1, -1, 0.02040816326530612) +
          STENCIL7_TAP(-1,  0, 0.02040816326530612) +
          STENCIL7_TAP(-1,  1, 0.02040816326530612) +
          STENCIL7_TAP(-1,  2, 0.02040816326530612) +
          STENCIL7_TAP(-1,  3, 0.02040816326530612) +
          STENCIL7_TAP( 0, -3, 0.02040816326530612) +
          STENCIL7_TAP( 0, -2, 0.02040816326530612) +
          STENCIL7_TAP( 0, -1, 0.02040816326530612) +
          STENCIL7_TAP( 0,  0, 0.02040816326530612) +
          STENCIL7_TAP( 0,  1, 0.02040816326530612) +
          STENCIL7_TAP( 0,  2, 0.02040816326530612) +
          STENCIL7_TAP( 0,  3, 0.02040816326530612) +
          STENCIL7_TAP( 1, -3, 0.02040816326530612) +
          STENCIL7_TAP( 1, -2, 0.02040816326530612) +
          STENCIL7_TAP( 1, -1, 0.02040816326530612) +
          STENCIL7_TAP( 1,  0, 0.02040816326530612) +
          STENCIL7_TAP( 1,  1, 0.02040816326530612) +
          STENCIL7_TAP( 1,  2, 0.02040816326530612) +
          STENCIL7_TAP( 1,  3, 0.02040816326530612) +
          STENCIL7_TAP( 2, -3, 0.02040816326530612) +
          STENCIL7_TAP( 2, -2, 0.02040816326530612) +
          STENCIL7_TAP( 2, -1, 0.02040816326530612) +
          STENCIL7_TAP( 2,  0, 0.02040816326530612) +
          STENCIL7_TAP( 2,  1, 0.02040816326530612) +
          STENCIL7_TAP( 2,  2, 0.02040816326530612) +
          STENCIL7_TAP( 2,  3, 0.02040816326530612) +
          STENCIL7_TAP( 3, -3, 0.02040816326530612) +
          STENCIL7_TAP( 3, -2, 0.02040816326530612) +
          STENCIL7_TAP( 3, -1, 0.02040816326530612) +
          STENCIL7_TAP( 3,  0, 0.02040816326530612) +
          STENCIL7_TAP( 3,  1, 0.02040816326530612) +
          STENCIL7_TAP( 3,  2, 0.02040816326530612) +
          STENCIL7_TAP( 3,  3, 0.02040816326530612);
#pragma endscop
}

#undef STENCIL7_TAP

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
