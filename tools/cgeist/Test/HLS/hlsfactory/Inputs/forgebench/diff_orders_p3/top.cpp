
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

data_t BRAM_1[64][128];
data_t BRAM_2[128][128];
data_t BRAM_3[64][128];
data_t BRAM_4[64][128];

void load_64_128_ap_fixed_16_5_(data_t input[64][128], data_t output[64][128])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_128_ap_fixed_16_5_(data_t input[128][128], data_t output[128][128])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_KJI FUNCTION
//////////////////////////////////////////
/*==== GEMM_KJI FUNCTION START ====*/
void gemm_kji(
    ap_fixed<16, 5> input_A[64][128],
    ap_fixed<16, 5> input_B[128][128],
    ap_fixed<16, 5> output[64][128]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=8 dim=1
#pragma HLS array_partition variable=output cyclic factor=8 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=2
#pragma HLS array_partition variable=output cyclic factor=16 dim=2

for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
for (int i = 0; i < 64; i++) {
#pragma HLS unroll factor=8
    output[i][k] = 0;
}
}


for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
for (int j = 0; j < 128; j++) {
#pragma HLS unroll factor=16
for (int i = 0; i < 64; i++) {
#pragma HLS unroll factor=8
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_KJI FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_KJI FUNCTION
//////////////////////////////////////////


void store_64_128_ap_fixed_16_5_(data_t input[64][128], data_t output[64][128])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[64][128], data_t DRAM_2[128][128], data_t DRAM_3[64][128], data_t DRAM_4[64][128])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    load_64_128_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_128_128_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_64_128_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_KJI
//////////////////////////////////////////
#pragma HLS array_partition variable=BRAM_1 cyclic factor=8 dim=1
#pragma HLS array_partition variable=BRAM_4 cyclic factor=8 dim=1
#pragma HLS array_partition variable=BRAM_2 cyclic factor=16 dim=1
#pragma HLS array_partition variable=BRAM_1 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_2 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_4 cyclic factor=16 dim=2

for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
for (int i = 0; i < 64; i++) {
#pragma HLS unroll factor=8
    BRAM_4[i][k] = 0;
}
}


for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
for (int j = 0; j < 128; j++) {
#pragma HLS unroll factor=16
for (int i = 0; i < 64; i++) {
#pragma HLS unroll factor=8
    BRAM_4[i][k] += BRAM_1[i][j] * BRAM_2[j][k];
}
}
}
//////////////////////////////////////////
// End: Inline implementation of GEMM_KJI
//////////////////////////////////////////

    store_64_128_ap_fixed_16_5_(BRAM_4, DRAM_4);
}