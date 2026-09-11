#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifndef N
#define N 128
#endif

#ifndef REPEAT
#define REPEAT 50
#endif

#ifndef WARMUP
#define WARMUP 10
#endif

void kernel_whisper_softmax_full(float out[N], float x[N]);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void init_input(float x[N]) {
  for (int i = 0; i < N; ++i) {
    float a = (float)((i * 17) % 97) * 0.03125f;
    float b = (float)((i * 7) % 19) * 0.015625f;
    x[i] = a - b - 1.5f;
  }
}

static double checksum(const float out[N]) {
  double s = 0.0;
  for (int i = 0; i < N; ++i)
    s += (double)out[i];
  return s;
}

int main(void) {
  static float x[N];
  static float out[N];
  init_input(x);

  for (int iter = 0; iter < WARMUP + REPEAT; ++iter) {
    double t0 = now_ms();
    kernel_whisper_softmax_full(out, x);
    double t1 = now_ms();
    if (iter >= WARMUP) {
      printf("WHISPER_TIMING\tbench=softmax_full\tN=%d\titer=%d\thost_ms=%.6f\n",
             N, iter - WARMUP, t1 - t0);
    }
  }

  printf("WHISPER_OUTPUT\tbench=softmax_full\tN=%d\tchecksum=%.9f\tfirst=%.9f\tlast=%.9f\n",
         N, checksum(out), (double)out[0], (double)out[N - 1]);
  return 0;
}
