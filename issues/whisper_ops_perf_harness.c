#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifndef N
#define N 128
#endif

#ifndef CONV_IN
#define CONV_IN 160
#endif

#ifndef CONV_K
#define CONV_K 3
#endif

#ifndef REPEAT
#define REPEAT 50
#endif

#ifndef WARMUP
#define WARMUP 10
#endif

#define CONV_OUT (CONV_IN - CONV_K + 1)

void kernel_whisper_vec_dot(float out[1], float x[N], float y[N]);
float kernel_whisper_vec_softmax(float out[N], float x[N], float max_val);
void kernel_whisper_softmax_full(float out[N], float x[N]);
void kernel_whisper_rms_norm(float out[N], float x[N], float eps);
void kernel_whisper_gelu(float out[N], float x[N]);
void kernel_whisper_conv1d(int n, int k, float *out, const float *x,
                           const float *filter);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void init_vec(float *x, int n, int seed) {
  for (int i = 0; i < n; ++i) {
    float a = (float)(((i + seed) * 17) % 97) * 0.03125f;
    float b = (float)(((i + seed) * 7) % 19) * 0.015625f;
    x[i] = a - b - 1.5f;
  }
}

static double checksum(const float *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += (double)x[i];
  return s;
}

int main(void) {
  static float x[N > CONV_IN ? N : CONV_IN];
  static float y[N > CONV_IN ? N : CONV_IN];
  static float out[N > CONV_OUT ? N : CONV_OUT];
  static float filter[CONV_K];
  float scalar = 0.0f;
  float ret = 0.0f;

  init_vec(x, N > CONV_IN ? N : CONV_IN, 1);
  init_vec(y, N > CONV_IN ? N : CONV_IN, 11);
  init_vec(filter, CONV_K, 23);

  for (int iter = 0; iter < WARMUP + REPEAT; ++iter) {
    double t0 = now_ms();
#if BENCH_KIND == 1
    kernel_whisper_vec_dot(&scalar, x, y);
#elif BENCH_KIND == 2
    ret = kernel_whisper_vec_softmax(out, x, 1.0f);
#elif BENCH_KIND == 3
    kernel_whisper_softmax_full(out, x);
#elif BENCH_KIND == 4
    kernel_whisper_rms_norm(out, x, 1.0e-5f);
#elif BENCH_KIND == 5
    kernel_whisper_gelu(out, x);
#elif BENCH_KIND == 6
    kernel_whisper_conv1d(CONV_IN, CONV_K, out, x, filter);
#else
#error "Define BENCH_KIND as 1..6"
#endif
    double t1 = now_ms();
    if (iter >= WARMUP) {
      printf("WHISPER_TIMING\tkind=%d\titer=%d\thost_ms=%.6f\n",
             BENCH_KIND, iter - WARMUP, t1 - t0);
    }
  }

#if BENCH_KIND == 1
  printf("WHISPER_OUTPUT\tkind=%d\tscalar=%.9f\n", BENCH_KIND, (double)scalar);
#elif BENCH_KIND == 2
  printf("WHISPER_OUTPUT\tkind=%d\tchecksum=%.9f\tret=%.9f\tfirst=%.9f\tlast=%.9f\n",
         BENCH_KIND, checksum(out, N), (double)ret, (double)out[0],
         (double)out[N - 1]);
#elif BENCH_KIND == 6
  printf("WHISPER_OUTPUT\tkind=%d\tchecksum=%.9f\tfirst=%.9f\tlast=%.9f\n",
         BENCH_KIND, checksum(out, CONV_OUT), (double)out[0],
         (double)out[CONV_OUT - 1]);
#else
  printf("WHISPER_OUTPUT\tkind=%d\tchecksum=%.9f\tfirst=%.9f\tlast=%.9f\n",
         BENCH_KIND, checksum(out, N), (double)out[0], (double)out[N - 1]);
#endif
  return 0;
}
