
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

data_t BRAM_1[16][8];
data_t BRAM_2[8][32];
data_t BRAM_3[16][32];
data_t BRAM_4[16][32];
data_t BRAM_5[32];
data_t BRAM_6[16];
data_t BRAM_7[16];
data_t BRAM_8[32];
data_t BRAM_9[32];
data_t BRAM_10[32];
data_t BRAM_11[1];
data_t BRAM_12[1];

void load_16_8_ap_fixed_16_5_(data_t input[16][8], data_t output[16][8])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 8; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_8_32_ap_fixed_16_5_(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
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
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[16][8],
    ap_fixed<16, 5> input_B[8][32],
    ap_fixed<16, 5> bias[16][32],
    ap_fixed<16, 5> output[16][32]
)

{
#pragma HLS array_partition variable=input_B cyclic factor=2 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=2 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=2
#pragma HLS array_partition variable=output cyclic factor=16 dim=2
#pragma HLS array_partition variable=bias cyclic factor=16 dim=2

for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=1
for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
    output[i][k] = bias[i][k];
}
}


for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=1
for (int j = 0; j < 8; j++) {
#pragma HLS unroll factor=2
for (int k = 0; k < 32; k++) {
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


//////////////////////////////////////////
// Begin: GEMM_KIJ_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_KIJ_BIAS FUNCTION START ====*/
void gemm_kij_bias(
    ap_fixed<16, 5> input_A[16][8],
    ap_fixed<16, 5> input_B[8][32],
    ap_fixed<16, 5> bias[16][32],
    ap_fixed<16, 5> output[16][32]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=8 dim=1
#pragma HLS array_partition variable=output cyclic factor=8 dim=1
#pragma HLS array_partition variable=bias cyclic factor=8 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=2 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=2 dim=2

for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=1
for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=8
    output[i][k] = bias[i][k];
}
}


for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=1
for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=8
for (int j = 0; j < 8; j++) {
#pragma HLS unroll factor=2
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_KIJ_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_KIJ_BIAS FUNCTION with BIAS
//////////////////////////////////////////


void store_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[16][8], data_t DRAM_2[8][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32], data_t DRAM_5[32], data_t DRAM_10[32], data_t DRAM_11[1], data_t DRAM_12[1])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2
    #pragma HLS interface m_axi port=DRAM_5 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_10 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_11 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_12 offset=slave bundle=mem2

    load_16_8_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_8_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_32_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////
#pragma HLS array_partition variable=BRAM_2 cyclic factor=2 dim=1
#pragma HLS array_partition variable=BRAM_1 cyclic factor=2 dim=2
#pragma HLS array_partition variable=BRAM_2 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_4 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_3 cyclic factor=16 dim=2

for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=1
for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
    BRAM_4[i][k] = BRAM_3[i][k];
}
}


for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=1
for (int j = 0; j < 8; j++) {
#pragma HLS unroll factor=2
for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
    BRAM_4[i][k] += BRAM_1[i][j] * BRAM_2[j][k];
}
}
}
//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    //////////////////////////////////////////
// Begin: Call to GEMM_IJK_BIAS
//////////////////////////////////////////
gemm_ijk_bias(BRAM_1, BRAM_2, BRAM_3, BRAM_4);
//////////////////////////////////////////
// End: Call to GEMM_IJK_BIAS
//////////////////////////////////////////

    //////////////////////////////////////////
// Begin: Call to GEMM_KIJ_BIAS
//////////////////////////////////////////
gemm_kij_bias(BRAM_1, BRAM_2, BRAM_3, BRAM_4);
//////////////////////////////////////////
// End: Call to GEMM_KIJ_BIAS
//////////////////////////////////////////

    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_KIJ_BIAS
//////////////////////////////////////////
#pragma HLS array_partition variable=BRAM_1 cyclic factor=8 dim=1
#pragma HLS array_partition variable=BRAM_4 cyclic factor=8 dim=1
#pragma HLS array_partition variable=BRAM_3 cyclic factor=8 dim=1
#pragma HLS array_partition variable=BRAM_2 cyclic factor=4 dim=1
#pragma HLS array_partition variable=BRAM_1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=BRAM_2 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_4 cyclic factor=16 dim=2
#pragma HLS array_partition variable=BRAM_3 cyclic factor=16 dim=2

for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=8
    BRAM_4[i][k] = BRAM_3[i][k];
}
}


for (int k = 0; k < 32; k++) {
#pragma HLS unroll factor=16
for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor=8
for (int j = 0; j < 8; j++) {
#pragma HLS unroll factor=4
    BRAM_4[i][k] += BRAM_1[i][j] * BRAM_2[j][k];
}
}
}
//////////////////////////////////////////
// End: Inline implementation of GEMM_KIJ_BIAS
//////////////////////////////////////////

    store_16_32_ap_fixed_16_5_(BRAM_4, DRAM_4);
}