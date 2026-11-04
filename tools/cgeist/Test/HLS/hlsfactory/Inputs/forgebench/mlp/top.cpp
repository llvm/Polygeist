
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

data_t INPUT_BATCH[256][64];
data_t LAYER_1[64][128];
data_t BIAS_1[256][128];
data_t ITM_BATCH[256][128];
data_t RELU_BATCH[256][128];
data_t LAYER_2[128][16];
data_t BIAS_2[256][16];
data_t ITM_2_BATCH[256][16];
data_t OUT_BATCH[256][16];

void load_256_64_ap_fixed_16_5_(data_t input[256][64], data_t output[256][64])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_128_ap_fixed_16_5_(data_t input[64][128], data_t output[64][128])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_256_128_ap_fixed_16_5_(data_t input[256][128], data_t output[256][128])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_16_ap_fixed_16_5_(data_t input[128][16], data_t output[128][16])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 16; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_256_16_ap_fixed_16_5_(data_t input[256][16], data_t output[256][16])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 16; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[256][64],
    ap_fixed<16, 5> input_B[64][128],
    ap_fixed<16, 5> bias[256][128],
    ap_fixed<16, 5> output[256][128]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=1
#pragma HLS array_partition variable=output cyclic factor=16 dim=1
#pragma HLS array_partition variable=bias cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=2
#pragma HLS array_partition variable=output cyclic factor=16 dim=2
#pragma HLS array_partition variable=bias cyclic factor=16 dim=2

for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
    output[i][k] = bias[i][k];
}
}


for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int j = 0; j < 64; j++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16, 5>
 * Tensor dimensions: [256][128]
 *
 * Available functions:
 *   - relu
 *   - leaky_relu
 *   - prelu
 *   - rrelu
 *   - thresholded_relu
 *   - relu6
 *   - sigmoid
 *   - tanh_act
 *   - elu
 *   - selu
 *   - gelu
 *   - swish
 *   - softmax
 */

void relu_256_128_ap_fixed_16_5_(
    data_t input[256][128],
    data_t output[256][128]
)
{
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 128; j++) {
            output[i][j] = (input[i][j] > 0) ? input[i][j] : (data_t)0;
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[256][128],
    ap_fixed<16, 5> input_B[128][16],
    ap_fixed<16, 5> bias[256][16],
    ap_fixed<16, 5> output[256][16]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=1
#pragma HLS array_partition variable=output cyclic factor=16 dim=1
#pragma HLS array_partition variable=bias cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=2
#pragma HLS array_partition variable=output cyclic factor=16 dim=2
#pragma HLS array_partition variable=bias cyclic factor=16 dim=2

for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor=16
    output[i][k] = bias[i][k];
}
}


for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int j = 0; j < 128; j++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 16; k++) {
#pragma HLS unroll factor=16
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////


/*
 * Auto-generated Activation Functions for CNNs (3D version)
 * 
 * Data type: ap_fixed<16, 5>
 * Tensor dimensions: [256][16]
 *
 * Available functions:
 *   - relu
 *   - leaky_relu
 *   - prelu
 *   - rrelu
 *   - thresholded_relu
 *   - relu6
 *   - sigmoid
 *   - tanh_act
 *   - elu
 *   - selu
 *   - gelu
 *   - swish
 *   - softmax
 */

void softmax_256_16_ap_fixed_16_5_(
    data_t input[256][16],
    data_t output[256][16]
)
{
    // Compute softmax along the channel dimension for each spatial location.
    for (int i = 0; i < 256; i++) {
        data_t sum = 0; // compute softmax rowise
        for (int j = 0; j < 16; j++) {
            output[i][j] = hls::exp(input[i][j]);
            sum += output[i][j];
        }

        for (int j = 0; j < 16; j++) {
            output[i][j] /= sum;
        }
    }
}

void store_256_16_ap_fixed_16_5_(data_t input[256][16], data_t output[256][16])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 16; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[256][64], data_t DRAM_2[64][128], data_t DRAM_3[256][128], data_t DRAM_4[128][16], data_t DRAM_5[256][16], data_t DRAM_6[256][16])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_5 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_6 offset=slave bundle=mem2

    load_256_64_ap_fixed_16_5_(DRAM_1, INPUT_BATCH);
    load_64_128_ap_fixed_16_5_(DRAM_2, LAYER_1);
    load_256_128_ap_fixed_16_5_(DRAM_3, BIAS_1);
    load_128_16_ap_fixed_16_5_(DRAM_4, LAYER_2);
    load_256_16_ap_fixed_16_5_(DRAM_5, BIAS_2);
    //////////////////////////////////////////
// Begin: Call to GEMM_IJK_BIAS
//////////////////////////////////////////
gemm_ijk_bias(INPUT_BATCH, LAYER_1, BIAS_1, ITM_BATCH);
//////////////////////////////////////////
// End: Call to GEMM_IJK_BIAS
//////////////////////////////////////////

    relu_256_128_ap_fixed_16_5_(ITM_BATCH, RELU_BATCH);
    //////////////////////////////////////////
// Begin: Call to GEMM_IJK_BIAS
//////////////////////////////////////////
gemm_ijk_bias(RELU_BATCH, LAYER_2, BIAS_2, ITM_2_BATCH);
//////////////////////////////////////////
// End: Call to GEMM_IJK_BIAS
//////////////////////////////////////////

    softmax_256_16_ap_fixed_16_5_(ITM_2_BATCH, OUT_BATCH);
    store_256_16_ap_fixed_16_5_(OUT_BATCH, DRAM_6);
}