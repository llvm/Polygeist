
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
using namespace std;

typedef ap_fixed<16, 5> data_t;

data_t BRAM_image_input[128][14][14];
data_t BRAM_conv_weight[128][128][3][3];
data_t BRAM_conv_bias[128];
data_t BRAM_buffer_1[128][14][14];

void load_128_14_14_ap_fixed_16_5_(data_t input[128][14][14], data_t output[128][14][14])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 14; idx1++) {
            for (int idx2 = 0; idx2 < 14; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void load_128_128_3_3_ap_fixed_16_5_(data_t input[128][128][3][3], data_t output[128][128][3][3])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            for (int idx2 = 0; idx2 < 3; idx2++) {
                for (int idx3 = 0; idx3 < 3; idx3++) {
                    output[idx0][idx1][idx2][idx3] = input[idx0][idx1][idx2][idx3];
                }
            }
        }
    }
}

void load_128_ap_fixed_16_5_(data_t input[128], data_t output[128])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        output[idx0] = input[idx0];
    }
}

/*
 * Auto-generated Convolution HLS Code (with optional bias and grouping)
 *
 * Dimensions:
 *   Input  : [128][14][14]
 *   Kernel : [128][128][3][3]
 *   Output : [128][14][14]
 *
 * Data type: ap_fixed<16, 5>
 *
 * If bias is included, a 1D bias array of size [128] is expected.
 * For group convolution, we assume:
 *   128 % groups == 0
 *   128 % groups == 0
 */

// Use a typedef for the data type

//////////////////////////////////////////
// Begin: CONV2D FUNCTION (co in the innermost loop)
//////////////////////////////////////////

void conv2d_128_128_14_14_14_14_3_1_1_ap_fixed_16_5__bias(
    data_t input[128][14][14],
    data_t kernel[128][128][3][3],
    data_t bias[128],
    data_t output[128][14][14]
)
{
    #pragma HLS array_partition variable=input  type=cyclic factor=8  dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=64 dim=1
    #pragma HLS array_partition variable=kernel type=cyclic factor=8 dim=2
    #pragma HLS array_partition variable=bias   type=cyclic factor=64   dim=1
    #pragma HLS array_partition variable=output type=cyclic factor=64  dim=1
    // Initialize output to bias[co], with co in the innermost loop
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int co = 0; co < 128; co++) {
            #pragma HLS unroll factor=64
                output[co][i][j] = bias[co];
            }
        }
    }

    // Perform convolution (co in the innermost loop)
    for (int i = 0; i < 14; i++) {
        for (int j = 0; j < 14; j++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_row = i * 1 - 1 + kh;
                    int in_col = j * 1 - 1 + kw;
                    if (in_row >= 0 && in_row < 14 && in_col >= 0 && in_col < 14) {
                        for (int ci = 0; ci < 128; ci++) {
                        #pragma HLS unroll factor=8
                            for (int co = 0; co < 128; co++) {
                                #pragma HLS unroll factor=64
                                output[co][i][j] += input[ci][in_row][in_col] * kernel[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
}

void store_128_14_14_ap_fixed_16_5_(data_t input[128][14][14], data_t output[128][14][14])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 14; idx1++) {
            for (int idx2 = 0; idx2 < 14; idx2++) {
                output[idx0][idx1][idx2] = input[idx0][idx1][idx2];
            }
        }
    }
}

void top(data_t DRAM_image_input[128][14][14], data_t DRAM_conv_weight[128][128][3][3], data_t DRAM_conv_bias[128], data_t DRAM_image_output[128][14][14])
{
    #pragma HLS interface m_axi port=DRAM_image_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_conv_weight offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_conv_bias offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_image_output offset=slave bundle=mem2

    load_128_14_14_ap_fixed_16_5_(DRAM_image_input, BRAM_image_input);
    load_128_128_3_3_ap_fixed_16_5_(DRAM_conv_weight, BRAM_conv_weight);
    load_128_ap_fixed_16_5_(DRAM_conv_bias, BRAM_conv_bias);
    conv2d_128_128_14_14_14_14_3_1_1_ap_fixed_16_5__bias(BRAM_image_input, BRAM_conv_weight, BRAM_conv_bias, BRAM_buffer_1);
    store_128_14_14_ap_fixed_16_5_(BRAM_buffer_1, DRAM_image_output);
}