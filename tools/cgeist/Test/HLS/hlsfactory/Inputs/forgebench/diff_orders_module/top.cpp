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

// LOADS A
void load_64_128_ap_fixed_16_5_(data_t input[64][128], data_t output[64][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_64_ap_fixed_16_5_(data_t input[128][64], data_t output[128][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_64_ap_fixed_16_5_(data_t input[64][64], data_t output[64][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

// LOADS B
void load_32_128_ap_fixed_16_5_(data_t input[32][128], data_t output[64][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_32_ap_fixed_16_5_(data_t input[128][32], data_t output[128][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_32_32_ap_fixed_16_5_(data_t input[32][32], data_t output[64][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

// LOADS C
void load_128_128_ap_fixed_16_5_(data_t input[128][128], data_t output[128][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void common_gemm_ikj(
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

    #pragma HLS inline off
for (int i = 0; i < 64; i++) {
    #pragma HLS unroll factor=8
    for (int k = 0; k < 128; k++) {
#pragma HLS unroll factor=16
output[i][k] = 0;
}
}


for (int i = 0; i < 64; i++) {
    #pragma HLS unroll factor=8
    for (int k = 0; k < 128; k++) {
    #pragma HLS unroll factor=16
    for (int j = 0; j < 128; j++) {
    #pragma HLS unroll factor=16
    output[i][k] += input_A[i][j] * input_B[j][k];
}
}
}
}


void store_64_64_ap_fixed_16_5_(data_t input[64][128], data_t output[64][64])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void store_32_32_ap_fixed_16_5_(data_t input[64][128], data_t output[32][32])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void store_64_128_ap_fixed_16_5_(data_t input[64][128], data_t output[64][128])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top_A(data_t DRAM_1[64][128], data_t DRAM_2[128][64], data_t DRAM_3[64][64], data_t DRAM_4[64][64])
{
    data_t BRAM_1[64][128];
    data_t BRAM_2[128][128];
    data_t BRAM_3[64][128];
    data_t BRAM_4[64][128];

    load_64_128_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_128_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_64_64_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_IJK
//////////////////////////////////////////
common_gemm_ikj(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Inline implementation of GEMM_IJK
//////////////////////////////////////////

    store_64_64_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_B(data_t DRAM_1[32][128], data_t DRAM_2[128][32], data_t DRAM_3[32][32], data_t DRAM_4[32][32])
{
    data_t BRAM_1[64][128];
    data_t BRAM_2[128][128];
    data_t BRAM_3[64][128];
    data_t BRAM_4[64][128];

    load_32_128_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_128_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_32_32_ap_fixed_16_5_(DRAM_3, BRAM_3);
//////////////////////////////////////////
// Begin: Inline implementation of GEMM_JKI
//////////////////////////////////////////
common_gemm_ikj(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Inline implementation of GEMM_JKI
//////////////////////////////////////////

    store_32_32_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_C(data_t DRAM_1[64][128], data_t DRAM_2[128][128], data_t DRAM_3[64][128], data_t DRAM_4[64][128])
{
    data_t BRAM_1[64][128];
    data_t BRAM_2[128][128];
    data_t BRAM_3[64][128];
    data_t BRAM_4[64][128];

    load_64_128_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_128_128_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_64_128_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of GEMM_KJI
//////////////////////////////////////////
common_gemm_ikj(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Inline implementation of GEMM_KJI
//////////////////////////////////////////

    store_64_128_ap_fixed_16_5_(BRAM_4, DRAM_4);
}


void top(
    data_t DRAM_A1[64][128], data_t DRAM_A2[128][64], data_t DRAM_A3[64][64], data_t DRAM_A4[64][64],
    data_t DRAM_B1[32][128], data_t DRAM_B2[128][32], data_t DRAM_B3[32][32], data_t DRAM_B4[32][32],
    data_t DRAM_C1[64][128], data_t DRAM_C2[128][128], data_t DRAM_C3[64][128], data_t DRAM_C4[64][128]
){

    #pragma HLS allocation function instances=load_64_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_128_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_64_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_32_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_128_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_32_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_128_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=common_gemm_ikj limit=1
    #pragma HLS allocation function instances=store_64_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=store_32_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=store_64_128_ap_fixed_16_5_ limit=1

    #pragma HLS interface m_axi port=DRAM_A1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A4 offset=slave bundle=mem2

    #pragma HLS interface m_axi port=DRAM_B1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B4 offset=slave bundle=mem2

    #pragma HLS interface m_axi port=DRAM_C1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_C2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_C3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_C4 offset=slave bundle=mem2

    // A
    data_t BRAM_A1[64][128];
    data_t BRAM_A2[128][128];
    data_t BRAM_A3[64][128];
    data_t BRAM_A4[64][128];

    load_64_128_ap_fixed_16_5_(DRAM_A1, BRAM_A1);
    load_128_64_ap_fixed_16_5_(DRAM_A2, BRAM_A2);
    load_64_64_ap_fixed_16_5_(DRAM_A3, BRAM_A3);
    common_gemm_ikj(BRAM_A1, BRAM_A2, BRAM_A4);
    store_64_64_ap_fixed_16_5_(BRAM_A4, DRAM_A4);

    // B
    data_t BRAM_B1[64][128];
    data_t BRAM_B2[128][128];
    data_t BRAM_B3[64][128];
    data_t BRAM_B4[64][128];

    load_32_128_ap_fixed_16_5_(DRAM_B1, BRAM_B1);
    load_128_32_ap_fixed_16_5_(DRAM_B2, BRAM_B2);
    load_32_32_ap_fixed_16_5_(DRAM_B3, BRAM_B3);
    common_gemm_ikj(BRAM_B1, BRAM_B2, BRAM_B4);
    store_32_32_ap_fixed_16_5_(BRAM_B4, DRAM_B4);

    // C
    data_t BRAM_C1[64][128];
    data_t BRAM_C2[128][128];
    data_t BRAM_C3[64][128];
    data_t BRAM_C4[64][128];

    load_64_128_ap_fixed_16_5_(DRAM_C1, BRAM_C1);
    load_128_128_ap_fixed_16_5_(DRAM_C2, BRAM_C2);
    load_64_128_ap_fixed_16_5_(DRAM_C3, BRAM_C3);
    common_gemm_ikj(BRAM_C1, BRAM_C2, BRAM_C4);
    store_64_128_ap_fixed_16_5_(BRAM_C4, DRAM_C4);
}