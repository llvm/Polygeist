#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <ap_int.h>

//#define USE_FLOAT

#ifdef USE_FLOAT

  typedef float InputFixed;

  // Types for weights
  typedef ap_int<1> Bit;
  typedef ap_int<2> TwoBit;

  typedef float KType;
  typedef float HType;

  typedef float NormOutput;
  typedef ap_int<14> ConvOutput;

#else

  // Quantized 32-bit input images in the range [-1,1]
  typedef ap_fixed<32,2, AP_RND> InputFixed;

  // Types for weights
  typedef ap_int<1> Bit;
  typedef ap_int<2> TwoBit;

  typedef ap_fixed<16,2> KType;
  typedef ap_fixed<16,4> HType;

  typedef ap_fixed<16,5> NormOutput;
  typedef ap_int<14> ConvOutput;

#endif

#endif
