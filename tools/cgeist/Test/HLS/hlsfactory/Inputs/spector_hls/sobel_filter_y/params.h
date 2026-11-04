#ifndef PARAMS_H_
#define PARAMS_H_

#define PRAGMA_SUB(x) _Pragma(#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

#define W 1920
#define H 1080

#define DIMX_PARTITION_FACTOR 8
#define DIMY_PARTITION_FACTOR 4

#define UNROLL_FACTOR 1

#define SUBDIM_Y 32
#endif
