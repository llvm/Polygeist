///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv.h
// Description: Header file for tiling-based synthesizable implementation of 
//              ResNet-50's first 7x7 convolution layer with HD input image.
//
// Note:        You are required to only modify the value of MARGIN in this
//              file. DO NOT CHANGE ANY OTHER DEFINES as this file will not 
//              be included in your submission.
//
//              This file should be identical in both Part B and Part C.
///////////////////////////////////////////////////////////////////////////////

#ifndef CONV_H_
#define CONV_H_

#include <iostream>
#include <ap_fixed.h>

//--------------------------------------------------------------------------
// Type conversions for simulation and synthesis
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<16,3> fm_t;
    typedef ap_fixed<16,3> wt_t;
#endif

//--------------------------------------------------------------------------
// Configuration of MobileNet-v2's first 3 x 3 convolution layer with HD input 
//--------------------------------------------------------------------------
#define IN_FM_DEPTH       3
#define IN_FM_HEIGHT      96
#define IN_FM_WIDTH       96

#define OUT_FM_DEPTH      32
#define OUT_FM_HEIGHT     48
#define OUT_FM_WIDTH      48

#define PADDING           1 
#define KERNEL_HEIGHT     3
#define KERNEL_WIDTH      3

//--------------------------------------------------------------------------
// Divide the input image into multiple tiles 
//--------------------------------------------------------------------------
#define TILE_HEIGHT       32
#define TILE_WIDTH        32

#define N_TILE_ROWS (int) IN_FM_HEIGHT / TILE_HEIGHT
#define N_TILE_COLS (int) IN_FM_WIDTH  / TILE_WIDTH
//--------------------------------------------------------------------------
// TODO: Modify the value of MARGIN based on the number of additional
//       rows and columns (belonging to adjacent tiles) required 
//       to implement a functionally-correct tiling-based convolution.
//--------------------------------------------------------------------------
#define MARGIN            2*PADDING //TODO LARTHAM3: how is margin different from padding?

//--------------------------------------------------------------------------
// Input tile buffer dimensions 
//--------------------------------------------------------------------------
#define IN_BUF_DEPTH      32
#define IN_BUF_HEIGHT     TILE_HEIGHT + MARGIN 
#define IN_BUF_WIDTH      TILE_WIDTH  + MARGIN 

//--------------------------------------------------------------------------
// Output tile buffer dimensions 
//--------------------------------------------------------------------------
#define OUT_BUF_DEPTH     32
#define OUT_BUF_HEIGHT    TILE_HEIGHT
#define OUT_BUF_WIDTH     TILE_WIDTH

//--------------------------------------------------------------------------
// Top-level Function Declaration
//--------------------------------------------------------------------------
void tiled_conv (
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],

    fm_t layer1_ifmap        [32][48][48],
    fm_t fixp_conv_ifmap_1_0 [32][48][48],
    fm_t fixp_conv_ifmap_2_0 [16][48][48],
    fm_t fixp_conv_ifmap_2_1 [96][48][48],
    fm_t fixp_conv_ifmap_2_2 [96][24][24],
 
    wt_t layer_weights_0   [32][3][3][3],
    wt_t layer_bias_0      [32],
    wt_t layer_weights_1_0 [32][1][3][3],
    wt_t layer_bias_1_0    [32],
    wt_t layer_weights_1_1 [16][32][1][1],
    wt_t layer_bias_1_1    [16],
    wt_t layer_weights_2_0 [96][16][1][1],
    wt_t layer_bias_2_0    [96],
    wt_t layer_weights_2_1 [1][96][3][3],
    wt_t layer_bias_2_1    [96],
    wt_t layer_weights_2_2 [24][96][1][1],
    wt_t layer_bias_2_2    [24],

    fm_t output_feature_map[24][24][24]
);

#endif
