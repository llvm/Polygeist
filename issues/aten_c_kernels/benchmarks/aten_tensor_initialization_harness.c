#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 4096
#endif

static int closef(float a, float b) {
  return fabsf(a - b) <= 2.0e-5f * (1.0f + fabsf(b));
}

#if defined(TEST_FILL)
void aten_fill(float, float *);
int main(void) {
  float *x = malloc((size_t)N * sizeof(float));
  aten_fill(3.25f, x);
  for (int i=0;i<N;i++) if (x[i] != 3.25f) return 1;
  puts("PASS tensor-init fill"); free(x); return 0;
}
#elif defined(TEST_SEQUENCE)
#ifndef TEST_FUNCTION
#define TEST_FUNCTION aten_arange_cpu
#endif
void TEST_FUNCTION(float, float, float *);
int main(void) {
  float *x = malloc((size_t)N * sizeof(float));
  TEST_FUNCTION(-2.0f, 0.125f, x);
  for (int i=0;i<N;i++) if (!closef(x[i], -2.0f+0.125f*i)) return 1;
  puts("PASS tensor-init sequence"); free(x); return 0;
}
#elif defined(TEST_EYE)
void aten_eye_cpu(float *);
int main(void) {
  float *x = malloc((size_t)N*N*sizeof(float)); aten_eye_cpu(x);
  for(int i=0;i<N;i++)for(int j=0;j<N;j++)if(x[i*N+j]!=(i==j))return 1;
  puts("PASS tensor-init eye"); free(x); return 0;
}
#elif defined(TEST_DIAGONAL)
void aten_fill_diagonal_cpu(float *, float);
int main(void) {
  float *x = malloc((size_t)N*N*sizeof(float)); for(int i=0;i<N*N;i++)x[i]=(float)i;
  aten_fill_diagonal_cpu(x, -7.0f);
  for(int i=0;i<N;i++)for(int j=0;j<N;j++)if(x[i*N+j]!=(i==j?-7.0f:(float)(i*N+j)))return 1;
  puts("PASS tensor-init diagonal"); free(x); return 0;
}
#elif defined(TEST_LOGSPACE)
void aten_logspace_cpu(float, float, float, float *);
int main(void) {
  float *x = malloc((size_t)N*sizeof(float)); aten_logspace_cpu(-1.0f, 2.0f, 2.0f, x);
  for(int i=0;i<N;i++)if(!closef(x[i],powf(2.0f,-1.0f+3.0f*i/(N-1))))return 1;
  puts("PASS tensor-init logspace"); free(x); return 0;
}
#elif defined(TEST_MASKED_FILL)
void aten_masked_fill_cpu(float *, int *, float);
int main(void) {
  float *x=malloc((size_t)N*sizeof(float));int*m=malloc((size_t)N*sizeof(int));
  for(int i=0;i<N;i++){x[i]=i*.25f;m[i]=(i%3)==0;} aten_masked_fill_cpu(x,m,9.0f);
  for(int i=0;i<N;i++)if(x[i]!=(m[i]?9.0f:i*.25f))return 1;
  puts("PASS tensor-init masked-fill"); free(m);free(x);return 0;
}
#else
#error select a TEST_* mode
#endif
