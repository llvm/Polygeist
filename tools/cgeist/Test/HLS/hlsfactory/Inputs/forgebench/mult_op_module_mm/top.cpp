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
void load_16_64_ap_fixed_16_5_(data_t input[16][64], data_t output[16][64])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_32_ap_fixed_16_5_(data_t input[64][32], data_t output[64][32])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_ap_fixed_16_5_(data_t input[64], data_t output[64])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_64_mat_32_ap_fixed_16_5_(data_t input[64], data_t output[64][32])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 1; idx1++) {
            output[idx0][idx1] = input[idx0];
        }
    }
}


void load_16_ap_fixed_16_5_(data_t input[16], data_t output[16])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

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

#pragma HLS inline off
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

void mmv_ij(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[16]
)
{
    #pragma HLS inline off
for (int i = 0; i < 16; i++) {
for (int j = 0; j < 64; j++) {
    output[i] = 0;
}
}


for (int i = 0; i < 16; i++) {
for (int j = 0; j < 64; j++) {
    output[i] += input_A[i][j] * input_B[j];
}
}
}

void dot_product(
    ap_fixed<16, 5> input_A[64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[1]
)
{
    #pragma HLS inline off
for (int i = 0; i < 64; i++) {
    output[0] = 0;
}


for (int i = 0; i < 64; i++) {
    output[0] += input_A[i] * input_B[i];
}
}

void transpose(
    ap_fixed<16, 5> input[64][64]
)
{
    #pragma HLS inline off
for (int i = 0; i < 64; i++) {
for (int j = 0; j < 64; j++) {
    input[i][j] = input[j][i];
}
}
}

void dot_with_gemm(
    ap_fixed<16, 5> input_A[64],
    ap_fixed<16, 5> input_B[64][32],
    ap_fixed<16, 5> output[1]
)

{
    #pragma HLS inline off
    data_t mat_a[16][64];
    data_t out[16][32];
    
    for (int i = 0; i < 16; i++) {
        mat_a[0][i] = input_A[i];
    }
    
    gemm_ijk(mat_a, input_B, out);
    output[0] = out[0][0];
}

void mmv_with_gemm(
    ap_fixed<16, 5> input_A[16][64],
    ap_fixed<16, 5> input_B[64][32],
    ap_fixed<16, 5> output[16]
)

{
    #pragma HLS inline off
    data_t out[16][32];
    gemm_ijk(input_A, input_B, out);
    output = out[0];
}


// STORES A
void store_16_32_ap_fixed_16_5_(data_t input[16][32], data_t output[16][32])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

// STORES B
void store_16_ap_fixed_16_5_(data_t input[16], data_t output[16])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        output[idx0] = input[idx0];
    }
}

// STORES C
void store_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

void top_A(data_t DRAM_1[16][64], data_t DRAM_2[64][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32])
{
    data_t BRAM_1[16][64];
    data_t BRAM_2[64][32];
    data_t BRAM_3[16][32];
    data_t BRAM_4[16][32];

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

void top_B(data_t DRAM_1[16][64], data_t DRAM_2[64], data_t DRAM_3[16], data_t DRAM_4[16])
{
    data_t BRAM_1[16][64];
    data_t BRAM_2[64][32];
    data_t BRAM_3[16];
    data_t BRAM_4[16];

    load_16_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_mat_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
    // Begin: Call to MMV_IJ
//////////////////////////////////////////
mmv_with_gemm(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to MMV_IJ
//////////////////////////////////////////

    store_16_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_C(data_t DRAM_1[64], data_t DRAM_2[64], data_t DRAM_3[1], data_t DRAM_4[1])
{
    data_t BRAM_1[64];
    data_t BRAM_2[64][32];
    data_t BRAM_3[1];
    data_t BRAM_4[1];

    load_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_mat_32_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_1_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
    // Begin: Call to DOT_PRODUCT
//////////////////////////////////////////
dot_with_gemm(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to DOT_PRODUCT
//////////////////////////////////////////

    store_1_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top(
    data_t DRAM_A1[16][64], data_t DRAM_A2[64][32], data_t DRAM_A3[16][32], data_t DRAM_A4[16][32],
    data_t DRAM_B1[16][64], data_t DRAM_B2[64], data_t DRAM_B3[16], data_t DRAM_B4[16],
    data_t DRAM_C1[64], data_t DRAM_C2[64], data_t DRAM_C3[1], data_t DRAM_C4[1]
){

    #pragma HLS allocation function instances=load_16_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_64_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_16_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_16_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_1_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=gemm_ijk limit=1
    #pragma HLS allocation function instances=mmv_with_gemm limit=1
    #pragma HLS allocation function instances=dot_with_gemm limit=1
    #pragma HLS allocation function instances=store_16_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=store_16_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=store_1_ap_fixed_16_5_ limit=1

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

    //TOP_A
    data_t BRAM_A1[16][64];
    data_t BRAM_A2[64][32];
    data_t BRAM_A3[16][32];
    data_t BRAM_A4[16][32];

    load_16_64_ap_fixed_16_5_(DRAM_A1, BRAM_A1);
    load_64_32_ap_fixed_16_5_(DRAM_A2, BRAM_A2);
    load_16_32_ap_fixed_16_5_(DRAM_A3, BRAM_A3);
    gemm_ijk(BRAM_A1, BRAM_A2, BRAM_A4);
    store_16_32_ap_fixed_16_5_(BRAM_A4, DRAM_A4);

    //TOP_B
    data_t BRAM_B1[16][64];
    data_t BRAM_B2[64][32];
    data_t BRAM_B3[16];
    data_t BRAM_B4[16];

    load_16_64_ap_fixed_16_5_(DRAM_B1, BRAM_B1);
    load_64_mat_32_ap_fixed_16_5_(DRAM_B2, BRAM_B2);
    load_16_ap_fixed_16_5_(DRAM_B3, BRAM_B3);
    data_t out[16][32];
    gemm_ijk(BRAM_B1, BRAM_B2, out);
    for (int i = 0; i < 16; i++) {
        BRAM_B4[i] = out[i][0];
    }
    store_16_ap_fixed_16_5_(BRAM_B4, DRAM_B4);

    //TOP_C
    data_t BRAM_C1[64];
    data_t BRAM_C2[64][32];
    data_t BRAM_C3[1];
    data_t BRAM_C4[1];

    load_64_ap_fixed_16_5_(DRAM_C1, BRAM_C1);
    load_64_mat_32_ap_fixed_16_5_(DRAM_C2, BRAM_C2);
    load_1_ap_fixed_16_5_(DRAM_C3, BRAM_C3);
    data_t mat_a[16][64];
    data_t out_C[16][32];
    
    for (int i = 0; i < 16; i++) {
        mat_a[0][i] = BRAM_C1[i];
    }
    
    gemm_ijk(mat_a, BRAM_C2, out_C);
    BRAM_C4[0] = out_C[0][0];
    store_1_ap_fixed_16_5_(BRAM_C4, DRAM_C4);
}