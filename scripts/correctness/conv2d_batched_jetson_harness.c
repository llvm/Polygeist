/* conv2d_batched_jetson_harness.c — Jetson harness for the extracted
 * batched conv2d kernel. Provides a main(), inits inputs to a
 * deterministic pattern, calls the renamed `_impl` function (the
 * cgeist-lowered LLVM-ABI form of kernel_conv2d_batched), checksums
 * the output for correctness validation.
 *
 * Compile-time shape:  -DB= -DIC= -DOC= -DH= -DW= -DKS=
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Match conv2d_batched.c's dataset macros so -DLARGE_DATASET / -DMINI_DATASET
 * propagated from the build script sets all shapes consistently here. */
#if defined(LARGE_DATASET)
# define B  32
# define IC 64
# define OC 64
# define H  56
# define W  56
# define KS 3
#elif defined(MINI_DATASET)
# define B  4
# define IC 8
# define OC 8
# define H  32
# define W  32
# define KS 3
#endif
#ifndef B
# define B 4
#endif
#ifndef IC
# define IC 8
#endif
#ifndef OC
# define OC 8
#endif
#ifndef H
# define H 32
#endif
#ifndef W
# define W 32
#endif
#ifndef KS
# define KS 3
#endif
#define OH (H - KS + 1)
#define OW (W - KS + 1)

/* MLIR convert-func-to-llvm expands each memref<...xf32> to an 11-arg
 * descriptor for rank-4 (basePtr, alignedPtr, offset, 4×size, 4×stride).
 * The kernel name in the lowered LLVM IR is `kernel_conv2d_batched_impl`
 * after the build script sed-renames the original symbol. */
extern void kernel_conv2d_batched_impl(
    /* A: ?x?x?x?xf32 */
    float *A_b,  float *A_a,  int64_t A_o,
    int64_t A_s0, int64_t A_s1, int64_t A_s2, int64_t A_s3,
    int64_t A_t0, int64_t A_t1, int64_t A_t2, int64_t A_t3,
    /* F: ?x?x?x?xf32 */
    float *F_b,  float *F_a,  int64_t F_o,
    int64_t F_s0, int64_t F_s1, int64_t F_s2, int64_t F_s3,
    int64_t F_t0, int64_t F_t1, int64_t F_t2, int64_t F_t3,
    /* O: ?x?x?x?xf32 */
    float *O_b,  float *O_a,  int64_t O_o,
    int64_t O_s0, int64_t O_s1, int64_t O_s2, int64_t O_s3,
    int64_t O_t0, int64_t O_t1, int64_t O_t2, int64_t O_t3);

extern void   polygeist_cublas_time_begin(void);
extern double polygeist_cublas_time_end_ms(void);

static void run_kernel(float *A, float *F, float *Bout) {
  polygeist_cublas_time_begin();
  kernel_conv2d_batched_impl(
      A, A, 0,
      (int64_t)B, (int64_t)IC, (int64_t)H, (int64_t)W,
      (int64_t)(IC*H*W), (int64_t)(H*W), (int64_t)W, 1,
      F, F, 0,
      (int64_t)OC, (int64_t)IC, (int64_t)KS, (int64_t)KS,
      (int64_t)(IC*KS*KS), (int64_t)(KS*KS), (int64_t)KS, 1,
      Bout, Bout, 0,
      (int64_t)B, (int64_t)OC, (int64_t)OH, (int64_t)OW,
      (int64_t)(OC*OH*OW), (int64_t)(OH*OW), (int64_t)OW, 1);
  double ms = polygeist_cublas_time_end_ms();
  fprintf(stderr,
      "POLYGEIST_TIMING: conv2d_batched B=%d IC=%d OC=%d H=%d W=%d K=%d  %.3f ms\n",
      B, IC, OC, H, W, KS, ms);
}

int main(void) {
  size_t nA = (size_t)B*IC*H*W,
         nF = (size_t)OC*IC*KS*KS,
         nO = (size_t)B*OC*OH*OW;
  float *A = (float *)malloc(nA * sizeof(float));
  float *F = (float *)malloc(nF * sizeof(float));
  float *O = (float *)malloc(nO * sizeof(float));
  if (!A || !F || !O) { fprintf(stderr, "alloc failed\n"); return 1; }

  /* Same init as conv2d_batched.c's init_array (modular pattern). */
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
          A[((size_t)b*IC + c)*H*W + (size_t)i*W + j] =
              (float)((b + c + i + j) % 17) / 17.0f;
  for (int oc = 0; oc < OC; ++oc)
    for (int c = 0; c < IC; ++c)
      for (int i = 0; i < KS; ++i)
        for (int j = 0; j < KS; ++j)
          F[((size_t)oc*IC + c)*KS*KS + (size_t)i*KS + j] =
              (float)((oc*3 + c*5 + i*7 + j) % 11) / 11.0f;
  memset(O, 0, nO * sizeof(float));

  run_kernel(A, F, O);

  /* Checksum + selective dump for diff vs CPU stub. */
  double sum = 0;
  for (size_t k = 0; k < nO; ++k) sum += O[k];
  fprintf(stderr, "CHECKSUM: %.6f over %zu elems\n", sum, nO);
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  for (size_t k = 0; k < nO; ++k) {
    if (k % 19 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.4f ", O[k]);
  }
  fprintf(stderr, "\n==END   DUMP_ARRAYS==\n");

  free(A); free(F); free(O);
  return 0;
}
