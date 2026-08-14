#include <math.h>

#ifndef POINTWISE_N
#define POINTWISE_N 4194304
#endif

void pointwise_generic(float x[POINTWISE_N], float y[POINTWISE_N],
                       float scale, float offset,
                       float out[POINTWISE_N]) {
#pragma scop
  for (int i = 0; i < POINTWISE_N; ++i)
    out[i] = tanhf((x[i] - y[i]) * scale) + offset;
#pragma endscop
}
