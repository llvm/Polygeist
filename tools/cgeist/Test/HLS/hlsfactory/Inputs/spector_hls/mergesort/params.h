#include <cmath>

#ifndef PARAMS_H_
#define PARAMS_H_

#define PRAGMA_SUB(x) _Pragma(#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

#define no_size 16
const int no_size_loop = no_size;
const int max_iter = (no_size / 2);
#define outer_unroll 2
#define inner_unroll1 2
#define inner_unroll2 2
#define merge_unroll 2
#define partition_factor 2

#endif
