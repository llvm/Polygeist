#define N 64

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define DATA_TYPE float
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x,y) powf(x,y)

void bicg(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N]);
