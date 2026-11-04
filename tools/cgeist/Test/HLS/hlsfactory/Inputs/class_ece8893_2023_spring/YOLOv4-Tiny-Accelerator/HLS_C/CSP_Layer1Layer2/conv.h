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
    typedef ap_fixed<16,6> fm_t;
    typedef ap_fixed<16,6> wt_t;
#endif


#define MAX_DEPTH 1024
#define MAX_HEIGHT 416
#define MAX_WIDTH 416

//Layer - 1 configuration    


//--------------------------------------------------------------------------
// Configuration of ResNet-50's first 7 x 7 convolution layer with HD input 
//--------------------------------------------------------------------------
#define IN_FM_DEPTH1       64
#define IN_FM_HEIGHT1      104
#define IN_FM_WIDTH1       104

#define OUT_FM_DEPTH1      64
#define OUT_FM_HEIGHT1     104
#define OUT_FM_WIDTH1      104

#define STRIDE1            1
#define PADDING1           1 
#define KERNEL_HEIGHT1     3
#define KERNEL_WIDTH1      3

//--------------------------------------------------------------------------
// Divide the input image into multiple tiles 
//--------------------------------------------------------------------------
#define TILE_HEIGHT1       13 
#define TILE_WIDTH1        13

#define N_TILE_ROWS1 (int) IN_FM_HEIGHT1 / TILE_HEIGHT1
#define N_TILE_COLS1 (int) IN_FM_WIDTH1  / TILE_WIDTH1

//--------------------------------------------------------------------------
// TODO: Modify the value of MARGIN based on the number of additional
//       rows and columns (belonging to adjacent tiles) required 
//       to implement a functionally-correct tiling-based convolution.
//--------------------------------------------------------------------------
#define MARGIN1            2

//--------------------------------------------------------------------------
// Input tile buffer dimensions 
//--------------------------------------------------------------------------
#define IN_BUF_DEPTH1      64
#define IN_BUF_HEIGHT1     TILE_HEIGHT1 + MARGIN1 
#define IN_BUF_WIDTH1      TILE_WIDTH1  + MARGIN1 

//--------------------------------------------------------------------------
// Output tile buffer dimensions 
//--------------------------------------------------------------------------
#define OUT_BUF_DEPTH1     64
#define OUT_BUF_HEIGHT1    TILE_HEIGHT1 / STRIDE1
#define OUT_BUF_WIDTH1     TILE_WIDTH1  / STRIDE1

//--------------------------------------------------------------------------
// Configuration of ResNet-50's first 7 x 7 convolution layer with HD input 
//--------------------------------------------------------------------------
#define IN_FM_DEPTH2       32
#define IN_FM_HEIGHT2     104
#define IN_FM_WIDTH2      104

#define OUT_FM_DEPTH2      32
#define OUT_FM_HEIGHT2     104
#define OUT_FM_WIDTH2      104

#define STRIDE2            1
#define PADDING2           1 
#define KERNEL_HEIGHT2     3
#define KERNEL_WIDTH2      3

//--------------------------------------------------------------------------
// Divide the input image into multiple tiles 
//--------------------------------------------------------------------------
#define TILE_HEIGHT2      13
#define TILE_WIDTH2       13

#define N_TILE_ROWS2 (int) IN_FM_HEIGHT2 / TILE_HEIGHT2
#define N_TILE_COLS2 (int) IN_FM_WIDTH2  / TILE_WIDTH2

//--------------------------------------------------------------------------
// TODO: Modify the value of MARGIN based on the number of additional
//       rows and columns (belonging to adjacent tiles) required 
//       to implement a functionally-correct tiling-based convolution.
//--------------------------------------------------------------------------
#define MARGIN2            3

//--------------------------------------------------------------------------
// Input tile buffer dimensions 
//--------------------------------------------------------------------------
#define IN_BUF_DEPTH2      32
#define IN_BUF_HEIGHT2     TILE_HEIGHT2 + MARGIN2 
#define IN_BUF_WIDTH2      TILE_WIDTH2  + MARGIN2 

//--------------------------------------------------------------------------
// Output tile buffer dimensions 
//--------------------------------------------------------------------------
#define OUT_BUF_DEPTH2     8
#define OUT_BUF_HEIGHT2    TILE_HEIGHT2 / STRIDE2
#define OUT_BUF_WIDTH2     TILE_WIDTH2  / STRIDE2

//--------------------------------------------------------------------------
// Top-level Function Declaration
//--------------------------------------------------------------------------
void tiled_conv_L1 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_weights[1024][1024][3][3],
    wt_t layer_bias[1024],
    fm_t output_feature_map[1024][416][416]
);

void load_input_tile_block_from_DRAM_L1 (
	fm_t in_fm_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT1][IN_BUF_WIDTH1],
	fm_t in_fm[1024][416][416],
	int ti,
	int tj,
	int d
);

void load_layer_params_from_DRAM_L1 (
    wt_t weight_buf[OUT_BUF_DEPTH1][IN_BUF_DEPTH1][KERNEL_HEIGHT1][KERNEL_WIDTH1],
    wt_t bias_buf[OUT_BUF_DEPTH1],                                             
    wt_t weights[1024][1024][3][3],
    wt_t bias[1024],
    int  kernel_group,
    int d
);

void conv_3x3_L1 (
    fm_t Y_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1], 
    fm_t X_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT1][IN_BUF_WIDTH1],
    wt_t W_buf[OUT_BUF_DEPTH1][IN_BUF_DEPTH1][KERNEL_HEIGHT1][KERNEL_WIDTH1],
    wt_t B_buf[OUT_BUF_DEPTH1]
);

void save_partial_output_L1 (
    fm_t partial_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_DEPTH1],
    fm_t output_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_DEPTH1],
    wt_t bias_buf[OUT_BUF_DEPTH1],
    int d
);


void store_output_tile_to_DRAM_L1 (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1], 
    int  ti,
    int  tj,
    int  kernel_group
);

void tiled_conv_L2 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_weights[1024][1024][3][3],
    wt_t layer_bias[1024],
    fm_t output_feature_map[1024][416][416]
);

void load_input_tile_block_from_DRAM_L2 (
	fm_t in_fm_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2],
	fm_t in_fm[1024][416][416],
	int ti,
	int tj,
	int d
);

void load_layer_params_from_DRAM_L2 (
    wt_t weight_buf[OUT_BUF_DEPTH2][IN_BUF_DEPTH2][KERNEL_HEIGHT2][KERNEL_WIDTH2],
    wt_t bias_buf[OUT_BUF_DEPTH2],                                             
    wt_t weights[1024][1024][3][3],
    wt_t bias[1024],
    int  kernel_group,
    int d
);
void conv_3x3_L2 (
    fm_t Y_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2],
    wt_t W_buf[OUT_BUF_DEPTH2][IN_BUF_DEPTH2][KERNEL_HEIGHT2][KERNEL_WIDTH2],
    wt_t B_buf[OUT_BUF_DEPTH2]
);


void save_partial_output_L2 (
    fm_t partial_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_DEPTH2],
    fm_t output_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_DEPTH2],
    wt_t bias_buf[OUT_BUF_DEPTH2],
    int d
    );

void store_output_tile_to_DRAM_L2 (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    int  ti,
    int  tj,
    int  kernel_group
);

void copy_output_to_input_L2 (
    fm_t input_feature_map[1024][416][416],
    fm_t output_feature_map[1024][416][416]
);




void yolov4_tiny (
		fm_t input_image[1024][104][104],        
		wt_t conv_layer_1_weights[1024][1024][3][3],
		wt_t conv_layer_1_bias[1024],
		wt_t conv_layer_2_weights[1024][1024][3][3],
		wt_t conv_layer_2_bias[1024],
		fm_t output_feature_map[1024][416][416],
		fm_t output_L1_feature_map[64][104][104],
		fm_t output_L2_feature_map[32][104][104],
		fm_t input_feature_map[1024][416][416]
		);

#endif
