
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

data_t BRAM_1[256][128];
data_t BRAM_2[128][192];
data_t BRAM_3[256][192];
data_t BRAM_4[256][192];

void load_256_128_ap_fixed_16_5_(data_t input[256][128], data_t output[256][128])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_192_ap_fixed_16_5_(data_t input[128][192], data_t output[128][192])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 192; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_256_192_ap_fixed_16_5_(data_t input[256][192], data_t output[256][192])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 192; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

//////////////////////////////////////////
// Begin: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////
/*==== GEMM_IJK_BIAS FUNCTION START ====*/
void gemm_ijk_bias(
    ap_fixed<16, 5> input_A[256][128],
    ap_fixed<16, 5> input_B[128][192],
    ap_fixed<16, 5> bias[256][192],
    ap_fixed<16, 5> output[256][192]
)

{
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=1
#pragma HLS array_partition variable=output cyclic factor=16 dim=1
#pragma HLS array_partition variable=bias cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=4 dim=1
#pragma HLS array_partition variable=input_A cyclic factor=4 dim=2
#pragma HLS array_partition variable=input_B cyclic factor=8 dim=2
#pragma HLS array_partition variable=output cyclic factor=8 dim=2
#pragma HLS array_partition variable=bias cyclic factor=8 dim=2

for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 192; k++) {
#pragma HLS unroll factor=8
    output[i][k] = bias[i][k];
}
}


for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int j = 0; j < 128; j++) {
#pragma HLS unroll factor=4
for (int k = 0; k < 192; k++) {
#pragma HLS unroll factor=8
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}
/*==== GEMM_IJK_BIAS FUNCTION END ====*/
//////////////////////////////////////////
// END: GEMM_IJK_BIAS FUNCTION with BIAS
//////////////////////////////////////////


void store_256_192_ap_fixed_16_5_(data_t input[256][192], data_t output[256][192])
{
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 192; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_1[256][128], data_t DRAM_2[128][192], data_t DRAM_3[256][192], data_t DRAM_4[256][192])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    load_256_128_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_128_192_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_256_192_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////
#pragma HLS array_partition variable=BRAM_1 cyclic factor=16 dim=1
#pragma HLS array_partition variable=BRAM_4 cyclic factor=16 dim=1
#pragma HLS array_partition variable=BRAM_3 cyclic factor=16 dim=1
#pragma HLS array_partition variable=BRAM_2 cyclic factor=4 dim=1
#pragma HLS array_partition variable=BRAM_1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=BRAM_2 cyclic factor=8 dim=2
#pragma HLS array_partition variable=BRAM_4 cyclic factor=8 dim=2
#pragma HLS array_partition variable=BRAM_3 cyclic factor=8 dim=2

for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int k = 0; k < 192; k++) {
#pragma HLS unroll factor=8
    BRAM_4[i][k] = BRAM_3[i][k];
}
}


for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16
for (int j = 0; j < 128; j++) {
#pragma HLS unroll factor=4
for (int k = 0; k < 192; k++) {
#pragma HLS unroll factor=8
    BRAM_4[i][k] += BRAM_1[i][j] * BRAM_2[j][k];
}
}
}
//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK_BIAS
//////////////////////////////////////////

    store_256_192_ap_fixed_16_5_(BRAM_4, DRAM_4);
}