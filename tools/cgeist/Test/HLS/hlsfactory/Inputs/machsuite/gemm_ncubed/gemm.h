//Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include "support.h"

//Define compute data type
#define TYPE double

//Specify row/column sizes
#define row_size 64
#define col_size 64
#define N row_size*col_size

//Define the input range to operate over
#define MIN 0.
#define MAX 1.0

//Set number of iterations to execute
#define MAX_ITERATION 1

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE m1[N];
  TYPE m2[N];
  TYPE prod[N];
};
