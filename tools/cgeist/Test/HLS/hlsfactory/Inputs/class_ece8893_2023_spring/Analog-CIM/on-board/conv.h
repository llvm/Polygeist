#ifndef CONV_H_
#define CONV_H_

#include <ap_fixed.h>

//typedef ap_fixed<21,6> f_t; // smallest that passes sim
//typedef ap_fixed<26,6> f_t; // 32b version
//typedef ap_fixed<64,6> f_t; // for on-board testing
typedef ap_fixed<64,11> f_t; // for on-board testing


// arguments that the user can change
#define INPUT_PRECISION 8
#define WEIGHT_PRECISION 8
#define PARALLEL_ROWS 21
#define ADC_PRECISION 5
#define ADC_LEVELS 32 // 2**ADC_PRECISION
#define NUM_ARGS 2
#define VDD_POS 0
#define RES_DIV_POS 1
// #define RES_DIVIDER 505.0995
// #define RES_DIVIDER 1552.9583
#define RES_DIVIDER 1352 //1351.7611
#define VDD 1 //1.0

// dimensions of inputs and weights for the first layer of resnet-50
#define IN_ROWS 12545
#define IN_COLS 147
#define WT_ROWS 147
#define WT_COLS 65
#define WT_BIN_COLS 65*8

//--------------------------------------------------------------------------
// Divide the input image into multiple tiles 
//--------------------------------------------------------------------------
#define TILE_HEIGHT       193
#define TILE_WIDTH        21 

#define N_TILE_ROWS (int) IN_ROWS / TILE_HEIGHT
#define N_TILE_COLS (int) IN_COLS  / TILE_WIDTH

//--------------------------------------------------------------------------
// Input tile buffer dimensions 
//--------------------------------------------------------------------------
#define IN_BUF_HEIGHT     TILE_HEIGHT 
#define IN_BUF_WIDTH      TILE_WIDTH 

//--------------------------------------------------------------------------
// Weight tile buffer dimensions
//--------------------------------------------------------------------------
#define WT_BUF_HEIGHT    TILE_WIDTH
#define WT_BUF_WIDTH     13*8

#define N_WT_ROWS (int) WT_ROWS / WT_BUF_HEIGHT
#define N_WT_COLS (int) WT_BIN_COLS / WT_BUF_WIDTH

//--------------------------------------------------------------------------
// Output tile buffer dimensions 
//--------------------------------------------------------------------------
#define OUT_BUF_HEIGHT    TILE_HEIGHT
#define OUT_BUF_WIDTH     13

#define N_OUT_COLS (int) WT_COLS / OUT_BUF_WIDTH

// top level function declaration
void tiled_cim_conv(
  int input2d[IN_ROWS][IN_COLS],
  f_t weight2d_cond[WT_ROWS][WT_BIN_COLS],
  f_t v_ref[ADC_LEVELS],
  int output[IN_ROWS][WT_COLS],
  int cim_args[NUM_ARGS]
);

#endif
