
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

data_t BRAM_1[96][512];
data_t BRAM_2[512][128];
data_t BRAM_3[96][128];
data_t BRAM_4[96][128];

void load_96_512_ap_fixed_16_5_(data_t input[96][512], data_t output[96][512])
{
    for (int idx0 = 0; idx0 < 96; idx0++) {
        for (int idx1 = 0; idx1 < 512; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_512_128_ap_fixed_16_5_(data_t input[512][128], data_t output[512][128])
{
    for (int idx0 = 0; idx0 < 512; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_96_128_ap_fixed_16_5_(data_t input[96][128], data_t output[96][128])
{
    for (int idx0 = 0; idx0 < 96; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[96][512],
    ap_fixed<16, 5> input_B[512][128],
    ap_fixed<16, 5> bias[96][128],
    ap_fixed<16, 5> output[96][128]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=6 dim=1
#pragma HLS array_partition variable=output cyclic factor=6 dim=1
#pragma HLS array_partition variable=bias cyclic factor=6 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=4 dim=2
#pragma HLS array_partition variable=output cyclic factor=4 dim=2
#pragma HLS array_partition variable=bias cyclic factor=4 dim=2

for (int i = 0; i < 96; i++) {
#pragma HLS unroll factor=6
for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=4
    output[i][k] = bias[i][k];
}
}


for (int i = 0; i < 96; i++) {
#pragma HLS unroll factor=6
for (int j = 0; j < 512; j++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=4
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////


void store_96_128_ap_fixed_16_5_(data_t input[96][128], data_t output[96][128])
{
    for (int idx0 = 0; idx0 < 96; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[96][512], data_t DRAM_2[512][128], data_t DRAM_3[96][128], data_t DRAM_4[96][128])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    load_96_512_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_512_128_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_96_128_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////
#pragma HLS array_partition variable=BRAM_1 cyclic factor=6 dim=1
#pragma HLS array_partition variable=BRAM_4 cyclic factor=6 dim=1
#pragma HLS array_partition variable=BRAM_3 cyclic factor=6 dim=1
#pragma HLS array_partition variable=BRAM_2 cyclic factor=16 dim=1
#pragma HLS array_partition variable=BRAM_1 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_2 cyclic factor=4 dim=2
#pragma HLS array_partition variable=BRAM_4 cyclic factor=4 dim=2
#pragma HLS array_partition variable=BRAM_3 cyclic factor=4 dim=2

for (int i = 0; i < 96; i++) {
#pragma HLS unroll factor=6
for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=4
    BRAM_4[i][k] = BRAM_3[i][k];
}
}


for (int i = 0; i < 96; i++) {
#pragma HLS unroll factor=6
for (int j = 0; j < 512; j++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=4
    BRAM_4[i][k] += BRAM_1[i][j] * BRAM_2[j][k];
}
}
}
//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    store_96_128_ap_fixed_16_5_(BRAM_4, DRAM_4);
}