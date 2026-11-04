/*===============================================================*/
/*                                                               */
/*                          typedefs.h                           */
/*                                                               */
/*              Constant definitions and typedefs.               */
/*                                                               */
/*===============================================================*/

#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__



// dataset information
const int NUM_FEATURES  = 1024;
const int NUM_SAMPLES   = 5000;
const int NUM_TRAINING  = 4500;
const int NUM_TESTING   = 500;
const int STEP_SIZE     = 60000; 
const int NUM_EPOCHS    = 5;
const int DATA_SET_SIZE = NUM_FEATURES * NUM_SAMPLES;

#ifdef OCL
  #include <string>
  // target device
  // change here to map to a different device
  const std::string TARGET_DEVICE = "xilinx_aws-vu9p-f1-04261818_dynamic_5_0";
#endif

// datatypes for accelerator

#ifndef SW
  #ifdef SDSOC
    // embedded platforms have less off-chip bandwidth
    #define VFTYPE_WIDTH  64
    #define VDTYPE_WIDTH  64
  #else
    // take advantage of the off-chip bandwidth of ocl platforms
    #define VFTYPE_WIDTH  512
    #define VDTYPE_WIDTH  512
  #endif

  #include "ap_int.h"
  #include "ap_fixed.h"
  // features / parameters
  #define FTYPE_TWIDTH 32
  #define FTYPE_IWIDTH 13
  typedef ap_fixed<FTYPE_TWIDTH,FTYPE_IWIDTH> FeatureType;
  typedef ap_uint<VFTYPE_WIDTH>               VectorFeatureType;
  const unsigned int F_VECTOR_SIZE = VFTYPE_WIDTH / FTYPE_TWIDTH;
  // training data
  #define DTYPE_TWIDTH 16
  #define DTYPE_IWIDTH 4
  typedef ap_fixed<DTYPE_TWIDTH,DTYPE_IWIDTH>  DataType;
  typedef ap_uint<VDTYPE_WIDTH>                VectorDataType;
  const unsigned int D_VECTOR_SIZE = VDTYPE_WIDTH / DTYPE_TWIDTH;
  // label
  #define LTYPE_WIDTH   8
  #define VLTYPE_WIDTH  32
  typedef ap_uint<LTYPE_WIDTH>                 LabelType;
  typedef ap_uint<VLTYPE_WIDTH>                VectorLabelType;
  const unsigned int L_VECTOR_SIZE = VLTYPE_WIDTH / LTYPE_WIDTH;
  
  // datatypes for the LUT
  #define LUTOUT_TWIDTH     12
  #define LUTOUT_IWIDTH     2
  #define LUTIN_TWIDTH      12
  #define LUTIN_IWIDTH      4
  typedef ap_ufixed<32, 20> TmpFixed; 
  typedef ap_uint<LUTIN_TWIDTH> IdxFixed; 
  typedef ap_fixed<LUTIN_TWIDTH, LUTIN_IWIDTH> LutInFixed;
  typedef ap_fixed<LUTOUT_TWIDTH, LUTOUT_IWIDTH> LutOutFixed;
#else
  // software version uses C++ built-in datatypes
  typedef float FeatureType;
  typedef float DataType;
  typedef unsigned char LabelType;
  // and uses math functions to compute sigmoid values
  // no need to declare special datatype for sigmoid
#endif

#define PAR_FACTOR 32

#endif
