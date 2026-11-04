#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
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
    typedef float fm_t;
    typedef float mk_t;
    
#else
    typedef ap_fixed<16,2> fm_t;
    typedef ap_fixed<16,2> wt_t;
    typedef ap_fixed<16,2> fm_t;
    typedef ap_int<2>  mk_t; // mask bits
#endif
//--------------------------------------------------------------------------
// Function Declarations
//--------------------------------------------------------------------------

#define IM_SIZE 32

#define IN_BUF_DEPTH 8
#define OUT_BUF_DEPTH 8

#define OUT_BUF_HEIGHT 8
#define OUT_BUF_WIDTH 8
#define IN_BUF_HEIGHT OUT_BUF_HEIGHT + 2
#define IN_BUF_WIDTH OUT_BUF_WIDTH + 2

#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 16


void tiled_conv(
    fm_t fixp_layer1_ifmap[3][IM_SIZE][IM_SIZE],        // input image
    fm_t fixp_layer2_ifmap[32][IM_SIZE][IM_SIZE],       // output of conv1
    fm_t fixp_layer3_ifmap[32][IM_SIZE][IM_SIZE],       // output of conv2
    fm_t fixp_layer4_ifmap[32][IM_SIZE/2][IM_SIZE/2],   // output of max_pool1
    fm_t fixp_layer5_ifmap[64][IM_SIZE/2][IM_SIZE/2],   // output of conv3
    fm_t fixp_layer6_ifmap[64][IM_SIZE/2][IM_SIZE/2],   // output of conv4
    fm_t fixp_layer7_ifmap[64][IM_SIZE/4][IM_SIZE/4],   // output of max_pool2
    fm_t fixp_layer8_ifmap[128],                        // output of fc1
    fm_t fixp_layer9_ifmap[128],                        // output of ReLU1
    fm_t fixp_layer10_ifmap[10],                        // output of fc2

    wt_t fixp_conv1_weights[32][3][3][3],
    wt_t fixp_conv1_bias[32],
    wt_t fixp_conv2_weights[32][32][3][3],
    wt_t fixp_conv2_bias[32],
    wt_t fixp_conv3_weights[64][32][3][3],
    wt_t fixp_conv3_bias[64],
    wt_t fixp_conv4_weights[64][64][3][3],
    wt_t fixp_conv4_bias[64],
    wt_t fixp_fc1_weights[128][4096],
    wt_t fixp_fc1_bias[128],
    wt_t fixp_fc2_weights[10][128],
    wt_t fixp_fc2_bias[10],

    fm_t fixp_grad1[128],
    fm_t fixp_grad2[128],
    fm_t fixp_grad3[4096],
    fm_t fixp_grad4[64][IM_SIZE/4][IM_SIZE/4],
    fm_t fixp_grad5[64][IM_SIZE/2][IM_SIZE/2],
    fm_t fixp_grad6[64][IM_SIZE/2][IM_SIZE/2],
    fm_t fixp_grad7[32][IM_SIZE/2][IM_SIZE/2],
    fm_t fixp_grad8[32][IM_SIZE][IM_SIZE],
    fm_t fixp_grad9[32][IM_SIZE][IM_SIZE],
    fm_t fixp_grad10[3][IM_SIZE][IM_SIZE],

    mk_t fixp_mask_maxpool2[64][IM_SIZE/4][IM_SIZE/4],
    mk_t fixp_mask_maxpool1[32][IM_SIZE/2][IM_SIZE/2]

);


// void tiled_conv_forward(

// );

#endif