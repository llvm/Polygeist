// Minimal stand-in for the Vivado HLS ap_cint.h (C arbitrary-precision
// types), for cgeist tests only: each intN / uintN is the smallest builtin
// integer type that holds N bits.
#ifndef CGEIST_TEST_AP_CINT_H
#define CGEIST_TEST_AP_CINT_H
#define CGEIST_AP_CINT(N, T)                                                   \
  typedef signed T int##N;                                                     \
  typedef unsigned T uint##N;
CGEIST_AP_CINT(1, char) CGEIST_AP_CINT(2, char) CGEIST_AP_CINT(3, char)
CGEIST_AP_CINT(4, char) CGEIST_AP_CINT(5, char) CGEIST_AP_CINT(6, char)
CGEIST_AP_CINT(7, char) CGEIST_AP_CINT(9, short) CGEIST_AP_CINT(10, short)
CGEIST_AP_CINT(11, short) CGEIST_AP_CINT(12, short) CGEIST_AP_CINT(13, short)
CGEIST_AP_CINT(14, short) CGEIST_AP_CINT(15, short) CGEIST_AP_CINT(17, int)
CGEIST_AP_CINT(18, int) CGEIST_AP_CINT(19, int) CGEIST_AP_CINT(20, int)
CGEIST_AP_CINT(24, int) CGEIST_AP_CINT(31, int) CGEIST_AP_CINT(33, long long)
CGEIST_AP_CINT(48, long long) CGEIST_AP_CINT(63, long long)
#undef CGEIST_AP_CINT
typedef signed char int8;
typedef unsigned char uint8;
typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned uint32;
typedef long long int64;
typedef unsigned long long uint64;
#endif
