//--------------------------------------------------------------------------
// Except minor height and width increment to IN_BUF_HEIGHT
// and IN_BUF_WIDTH respectively, no other dimension should be modified.
//--------------------------------------------------------------------------

#ifndef CONV_H_
#define CONV_H_

#include <iostream>
#include <ap_fixed.h>

//--------------------------------------------------------------------------
// Compiler Defines
//--------------------------------------------------------------------------
#ifdef HLS_SIM
    #include "config.h"
#endif

//--------------------------------------------------------------------------
// Type Conversions
//--------------------------------------------------------------------------
#ifdef  CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
    typedef ap_fixed<16,2> fm_t;
    typedef ap_fixed<16,2> wt_t;
#endif

#define STRIDE             1
#define PADDING            1 

#define IN_BUF_DEPTH1      3
#define IN_BUF_DEPTH2      8
#define IN_BUF_HEIGHT     28 // Hint: 23 + ?? (how many border features are needed?) 
#define IN_BUF_WIDTH      28 // Hint: 20 + ?? (how many border features are needed?)
#define IN_BUF_HEIGHT2    15 // Hint: 23 + ?? (how many border features are needed?) 
#define IN_BUF_WIDTH2     15 // Hint: 20 + ?? (how many border features are needed?)

#define OUT_BUF_DEPTH     8
#define OUT_BUF_HEIGHT    26
#define OUT_BUF_WIDTH     26
#define OUT_BUF_HEIGHT2    13
#define OUT_BUF_WIDTH2     13

#define US_BUF_DEPTH 8
#define US_BUF_HEIGHT 13
#define US_BUF_WIDTH 13

#define N_TILE_ROWS (int) (416/OUT_BUF_HEIGHT)
#define N_TILE_COLS (int) (416/OUT_BUF_WIDTH)

#define MAX_INPUT_DEPTH 1024

//--------------------------------------------------------------------------
// Function Declarations
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM_id3 (
    fm_t in_fm_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[1024][416][416], 
    int  ti, 
    int  tj, 
    int  d
);

void load_maxpool_input_tile_block_from_DRAM(
    fm_t maxpool_in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2+1][OUT_BUF_WIDTH2+1],
    fm_t input_mp_feature_map[1024][416][416],
    int d
);

void load_upsample_input_tile_block_from_DRAM(
    fm_t upsample_in_buf[US_BUF_DEPTH][US_BUF_HEIGHT][US_BUF_WIDTH],
    fm_t input_us_feature_map[1024][416][416],
    int d
);

void load_layer_params_from_DRAM_id3 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH1][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
);

void load_layer_params_from_DRAM_id3_0 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH1][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
);

void load_layer_params_from_DRAM_id3_1 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH1][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
);


void conv_3x3_id3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH1][3][3]
);

void load_input_tile_block_from_DRAM_id16 (
    fm_t in_fm_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[1024][416][416], 
    int  ti, 
    int  tj, 
    int  d
);

void load_input_tile_block_from_DRAM_conv (
    fm_t in_fm_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2], 
    fm_t in_fm[1024][416][416], 
    int  ti, 
    int  tj, 
    int  d
);

void load_layer_params_from_DRAM_id16 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
);

void load_layer_params_from_DRAM_id16_0 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
);

void load_layer_params_from_DRAM_id16_1 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
);

void load_layer_params_from_DRAM_conv1x1 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1],
    wt_t weights[1024][1024][1][1],
    int b,
    int d
);

void load_layer_params_from_DRAM_bias (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[1024][1024][1][1],
    wt_t bias[255],
    int b,
    int d
);

void conv_3x3_id16 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3]
);

void conv_3x3_id16_0 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3]
);

void conv_3x3_id16_1 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3]
);

void conv_ver3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3]
);

void conv_1x1 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1]
);

void save_partial_output_tile_block (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    int  d
);

void save_partial_output_tile_block_conv (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
    int  d
);

void save_partial_output_tile_block_bias (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
    wt_t bias_buf[OUT_BUF_DEPTH],
    int d
);

void store_output_tile_to_DRAM (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    int  ti,
    int  tj,
    int  b
);

void store_output_tile_to_DRAM_ver1 (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b
);

void store_maxpool_output_tile_to_DRAM (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2], 
    int  ti,
    int  tj,
    int  b
);

void store_maxpool_output_tile_to_DRAM_stride1(
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
    int d
);

void store_upsample_output_tile_to_DRAM(
    fm_t out_fm[1024][416][416],
    fm_t out_fm_buf[US_BUF_DEPTH][US_BUF_HEIGHT*2][US_BUF_WIDTH*2],
    int d
);

void model_conv_bn (
    fm_t *input_feature_map,
    wt_t *layer_conv_weights,
    wt_t *layer_bn_weights,
    fm_t *output_feature_map,
    int input_d,
    int input_h,
    int input_w,
    int filter_size,
    int kernel_h,
    int kernel_w
);

void model_conv (
    fm_t *input_feature_map,
    wt_t *layer_conv_weights,
    wt_t *layer_bias,
    fm_t *output_feature_map,
    int input_d,
    int input_h,
    int input_w,
    int filter_size,
    int kernel_h,
    int kernel_w
);

void model_maxpool2D(
    fm_t *in_buf,
    fm_t *out_buf,
    int input_d,
    int input_h,
    int input_w,
    int stride
);

void model_upsample(
    fm_t *in_buf,
    fm_t *out_buf,
    int input_d,
    int input_h,
    int input_w
);

void upsample_2D(
    fm_t in_buf[US_BUF_DEPTH][US_BUF_HEIGHT][US_BUF_WIDTH],
    fm_t out_buf[US_BUF_DEPTH][US_BUF_HEIGHT*2][US_BUF_WIDTH*2]
);

void max_pool_2D(
    fm_t in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2]
);

void max_pool_2D_stride1(
    fm_t in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2+1][OUT_BUF_WIDTH2+1],
    fm_t out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2]
);

void yolov3_tiny (
    fm_t input_image[3][416][416],
    wt_t conv_layer_1_weights[1024][1024][3][3],
    wt_t conv_layer_2_weights[1024][1024][3][3],
    wt_t conv_layer_3_weights[1024][1024][3][3],
    wt_t conv_layer_4_weights[1024][1024][3][3],
    wt_t conv_layer_5_weights[1024][1024][3][3],
    wt_t conv_layer_6_weights[1024][1024][3][3],
    wt_t conv_layer_7_weights[1024][1024][3][3],
    wt_t conv_layer_8_weights[1024][1024][1][1],
    wt_t conv_layer_9_weights[1024][1024][3][3],
    wt_t conv_layer_10_weights[1024][1024][1][1],
    wt_t conv_layer_11_weights[1024][1024][1][1],
    wt_t conv_layer_12_weights[1024][1024][3][3],
    wt_t conv_layer_13_weights[1024][1024][1][1],
    wt_t bias_layer_10[255],
    wt_t bias_layer_13[255],
    fm_t input_feature_map[1024][416][416],
    fm_t output_feature_map[1024][416][416],
    fm_t output13_feature_map[1024][416][416],
    fm_t output8_feature_map[1024][416][416]
);

void copy_output_to_input(
    fm_t input_feature_map[1024][416][416],
    fm_t output_feature_map[1024][416][416],
    int d,
    int h,
    int w
);

// void max_pool_2D(
//     fm_t in_buf[16][416][416],
//     fm_t out_buf[16][208][208],
//     int max
// );

#endif
