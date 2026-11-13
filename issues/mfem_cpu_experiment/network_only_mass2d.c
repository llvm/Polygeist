/* Projection experiment: the whole MFEM mass-2D operator as ONE tensor-network
 * call (what LoopLibrarian would emit if the final `Y +=` stage were matched),
 *   Y[e,dy,dx] += sum Bt[dy,qy] Bt[dx,qx] D[e,qy,qx] B[qy,dy'] B[qx,dx'] X[e,dy',dx']
 * executed by runtime/polygeist_tensornet_cpu.c, compared with the normalised
 * MFEM C on identical inputs (full-output check + timing).  The network call is
 * written by hand here; runtime and libraries are exactly those of the
 * compiled experiment. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 1024
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 200
#endif
enum { D1D = 4, Q1D = 5, NE = MFEM_BENCH_NE };

void mfem_pa_mass_apply_2d_stage_sliced(const double *B, const double *Bt, const double *D,
                                        const double *X, double *Y);
void polygeist_cutensornet_network_f64(const int64_t *pointers, const int64_t *metadata);

static double B[Q1D * D1D], Bt[D1D * Q1D], Dq[NE * Q1D * Q1D], X[NE * D1D * D1D];
static double Yref[NE * D1D * D1D], Ynet[NE * D1D * D1D];

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double value(int i, int salt) { return (double)(((i * 17 + salt * 13 + 5) % 101) - 50) / 257.0; }

/* modes: 0=e 1=qy 2=dx 4=qx 8=dy' 9=dx' 10=dy */
static int64_t md[3 + 7 + 3 * 16];
static int64_t ptr[7];
static void build_network(void) {
  int c = 0;
  md[c++] = 1; md[c++] = 6; md[c++] = 1;          /* version, inputs, accumulate */
  const int64_t ranks[7] = {2, 2, 3, 2, 2, 3, 3};
  for (int i = 0; i < 7; ++i) md[c++] = ranks[i];
#define DIM(ext, str, mode) do { md[c++] = (ext); md[c++] = (str); md[c++] = (mode); } while (0)
  DIM(D1D, Q1D, 10); DIM(Q1D, 1, 1);                              /* Bt[dy][qy]  */
  DIM(D1D, Q1D, 2);  DIM(Q1D, 1, 4);                              /* Bt[dx][qx]  */
  DIM(NE, Q1D * Q1D, 0); DIM(Q1D, Q1D, 1); DIM(Q1D, 1, 4);        /* D[e][qy][qx] */
  DIM(Q1D, D1D, 1);  DIM(D1D, 1, 8);                              /* B[qy][dy']  */
  DIM(Q1D, D1D, 4);  DIM(D1D, 1, 9);                              /* B[qx][dx']  */
  DIM(NE, D1D * D1D, 0); DIM(D1D, D1D, 8); DIM(D1D, 1, 9);        /* X[e][dy'][dx'] */
  DIM(NE, D1D * D1D, 0); DIM(D1D, D1D, 10); DIM(D1D, 1, 2);       /* Y[e][dy][dx] */
  ptr[0] = (int64_t)(uintptr_t)Bt; ptr[1] = (int64_t)(uintptr_t)Bt;
  ptr[2] = (int64_t)(uintptr_t)Dq; ptr[3] = (int64_t)(uintptr_t)B;
  ptr[4] = (int64_t)(uintptr_t)B;  ptr[5] = (int64_t)(uintptr_t)X;
  ptr[6] = (int64_t)(uintptr_t)Ynet;
}

int main(void) {
  for (int i = 0; i < Q1D * D1D; ++i) B[i] = value(i, 1);
  for (int d = 0; d < D1D; ++d)
    for (int q = 0; q < Q1D; ++q) Bt[d * Q1D + q] = B[q * D1D + d];
  for (int i = 0; i < NE * Q1D * Q1D; ++i) Dq[i] = value(i, 3);
  for (int i = 0; i < NE * D1D * D1D; ++i) X[i] = value(i, 4);
  build_network();

  /* correctness: one application on zeroed outputs */
  memset(Yref, 0, sizeof(Yref)); memset(Ynet, 0, sizeof(Ynet));
  mfem_pa_mass_apply_2d_stage_sliced(B, Bt, Dq, X, Yref);
  polygeist_cutensornet_network_f64(ptr, md);
  double max_abs = 0.0;
  for (int i = 0; i < NE * D1D * D1D; ++i) {
    const double d = fabs(Yref[i] - Ynet[i]);
    if (d > max_abs) max_abs = d;
  }
  printf("correctness max_abs=%.3g %s\n", max_abs, max_abs < 1e-12 ? "PASS" : "FAIL");

  for (int w = 0; w < 5; ++w) { mfem_pa_mass_apply_2d_stage_sliced(B, Bt, Dq, X, Yref); polygeist_cutensornet_network_f64(ptr, md); }
  double t0 = now();
  for (int it = 0; it < BENCH_ITERS; ++it) mfem_pa_mass_apply_2d_stage_sliced(B, Bt, Dq, X, Yref);
  const double c_us = (now() - t0) * 1e6 / BENCH_ITERS;
  t0 = now();
  for (int it = 0; it < BENCH_ITERS; ++it) polygeist_cutensornet_network_f64(ptr, md);
  const double n_us = (now() - t0) * 1e6 / BENCH_ITERS;
  printf("plain_c_us=%.2f network_us=%.2f speedup=%.2fx\n", c_us, n_us, c_us / n_us);
  return max_abs < 1e-12 ? 0 : 1;
}
