/*===============================================================*/
/*                                                               */
/*                          typedefs.h                           */
/*                                                               */
/*                     Typedefs for the host                     */
/*                                                               */
/*===============================================================*/

#ifndef __HAAR_H__
#define __HAAR_H__

// constants
const int IMAGE_HEIGHT = 240;
const int IMAGE_WIDTH = 320;
const int RESULT_SIZE = 100;
const int IMAGE_MAXGREY = 255;
const int IMAGE_SIZE = ( IMAGE_HEIGHT * IMAGE_WIDTH );
const int TOTAL_NODES = 2913;
const int TOTAL_STAGES = 25;
const int TOTAL_COORDINATES = TOTAL_NODES * 12;
const int TOTAL_WEIGHTS = TOTAL_NODES * 3;
const int WINDOW_SIZE = 25;
const int SQ_SIZE = 2;
const int PYRAMID_HEIGHT = 12;
const int ROWS = 25;
const int COLS = 25;
const int NUM_BANKS = 12;
const int SIZE = 2913;

#ifdef OCL
  #include <string>
  // target device
  // change here to map to a different device
  const std::string TARGET_DEVICE = "xilinx_aws-vu9p-f1-04261818_dynamic_5_0";
#endif


#ifndef SW
  #include "ap_int.h"
  // HLS-specific datatypes
  typedef ap_uint<18> uint18_t;
  typedef ap_uint<10> uint10_t;
  typedef ap_uint<1>  bit;
  typedef ap_uint<5> uint5_t;
  typedef ap_uint<13> int_I;  
  typedef ap_uint<21> int_SI;
  typedef ap_uint<18> int_II;             
  typedef ap_uint<26> int_SII;
#endif

// standard datatypes
typedef struct MyPoint
{
  int x;
  int y;
} MyPoint;

typedef struct
{
  int width;
  int height;
} MySize;

typedef struct
{
  int x; 
  int y;
  int width;
  int height;
} MyRect;

typedef struct 
{
  int width;
  int height;
  int maxgrey;
  int flag;
} MyInputImage;

#endif
