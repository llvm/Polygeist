
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

data_t BRAM_1[16][64];
data_t BRAM_2[64][32];
data_t BRAM_3[16][32];
data_t BRAM_4[16][32];

void load_16_64_ap_fixed_16_5_(data_t input[16][64], data_t output[16][64])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_32_ap_fixed_16_5_(data_t input[64][32], data_t output[64][32])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_IJK FUNCTION
//////////////////////////////////////////
/*==== GEMM_IJK FUNCTION START ====*/
void gemm_ijk(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64][32],
    ap_fixed<16, 5> output[16][32]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=8 dim=1
#pragma HLS array_partition variable=output cyclic factor=8 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=2
#pragma HLS array_partition variable=output cyclic factor=16 dim=2

for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=8
for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
    output[i][k] = 0;
}
}


for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=8
for (int j = 0; j < 64; j++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK FUNCTION
//////////////////////////////////////////


void store_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[16][64], data_t DRAM_2[64][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    load_16_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_32_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Call to GEMM_IJK
//////////////////////////////////////////
gemm_ijk(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to GEMM_IJK
//////////////////////////////////////////

    store_16_32_ap_fixed_16_5_(BRAM_4, DRAM_4);
}