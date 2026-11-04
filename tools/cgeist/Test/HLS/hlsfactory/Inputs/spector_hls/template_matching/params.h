#ifndef PARAMS_H
#define PARAMS_H

#define PRAGMA_SUB(x) _Pragma(#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

// inhei * inwid // tmphei * tmpwid

#define indim 200
#define tmpdim 10
#define thre 1
#define size indim *indim
#define tmpsize tmpdim *tmpdim
#define UNROLL_FACTOR 1
#define UNROLL_LOOP1 1
#define UNROLL_LOOP2 1
#define UNROLL_LOOP3 1
#define UNROLL_LOOP4 1

#endif
