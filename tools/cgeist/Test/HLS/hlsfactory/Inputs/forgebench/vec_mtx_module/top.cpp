
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

data_t BRAM_A1[64][64];
data_t BRAM_A2[64];
data_t BRAM_A3[16];
data_t BRAM_A4[16];

data_t BRAM_B1[64][64];
data_t BRAM_B2[64];
data_t BRAM_B3[16];
data_t BRAM_B4[16];

void load_16_64_ap_fixed_16_5_(data_t input[16][64], data_t output[64][64])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
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

void load_16_ap_fixed_16_5_(data_t input[16], data_t output[16])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_64_16_ap_fixed_16_5_(data_t input[64][16], data_t output[64][64])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 16; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
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

//////////////////////////////////////////
// Begin: MMV_IJ FUNCTION
//////////////////////////////////////////
/*==== MMV_IJ FUNCTION START ====*/
void mmv_ij(
    ap_fixed<16, 5> input_A[64][64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[16]
)
{
    #pragma HLS array_partition variable=input_A cyclic factor=16 dim=1
    #pragma HLS array_partition variable=output cyclic factor=16 dim=1
    #pragma HLS array_partition variable=input_B cyclic factor=16 dim=1
    #pragma HLS array_partition variable=input_A cyclic factor=16 dim=2
    
    #pragma HLS inline off
    for (int i = 0; i < 16; i++) {
    #pragma HLS unroll factor=16
        output[i] = 0;
    }
    
    
    for (int i = 0; i < 16; i++) {
    #pragma HLS unroll factor=16
    for (int j = 0; j < 64; j++) {
    #pragma HLS unroll factor=16
        output[i] += input_A[i][j] * input_B[j];
    }
    }
    }
/*==== MMV_IJ FUNCTION END ====*/
//////////////////////////////////////////
// END: MMV_IJ FUNCTION
//////////////////////////////////////////

void vmm_with_mmv_ij(
    ap_fixed<16, 5> input_A[64][64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[16]
)
{
    #pragma HLS inline off
    transpose(input_A); // Transpose the input matrix A
    mmv_ij(input_A, input_B, output); // Call the MMV_IJ function
}


void store_16_ap_fixed_16_5_(data_t input[16], data_t output[16])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 16; idx0++) {
        output[idx0] = input[idx0];
    }
}

void top_A(data_t DRAM_1[16][64], data_t DRAM_2[64], data_t DRAM_3[16], data_t DRAM_4[16])
{
    data_t BRAM_1[64][64];
    data_t BRAM_2[64];
    data_t BRAM_3[16];
    data_t BRAM_4[16];

    load_16_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
// Begin: Inline implementation of MMV_IJ
//////////////////////////////////////////
mmv_ij(BRAM_1, BRAM_2, BRAM_4); // Call the MMV_IJ function

    // The output of MMV_IJ is stored in BRAM_3, which is then written to DRAM_4
//////////////////////////////////////////
// End: Inline implementation of MMV_IJ
//////////////////////////////////////////

    store_16_ap_fixed_16_5_(BRAM_4, DRAM_4);
}

void top_B(data_t DRAM_1[64][16], data_t DRAM_2[64], data_t DRAM_3[16], data_t DRAM_4[16])
{
    data_t BRAM_1[64][64];
    data_t BRAM_2[64];
    data_t BRAM_3[16];
    data_t BRAM_4[16];

    load_64_16_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_16_ap_fixed_16_5_(DRAM_3, BRAM_3);
    /////////////////////////////////////////
vmm_with_mmv_ij(BRAM_1, BRAM_2, BRAM_4); // Call the VMM_IJ function

    store_16_ap_fixed_16_5_(BRAM_4, DRAM_4); // Store the result in DRAM_4
}

void top(data_t DRAM_A1[16][64], data_t DRAM_A2[64], data_t DRAM_A3[16], data_t DRAM_A4[16],
          data_t DRAM_B1[64][16], data_t DRAM_B2[64], data_t DRAM_B3[16], data_t DRAM_B4[16])
{

    #pragma HLS allocation function instances=load_16_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_64_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_16_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_64_16_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=transpose limit=1
    #pragma HLS allocation function instances=mmv_ij limit=1
    #pragma HLS allocation function instances=vmm_with_mmv_ij limit=1
    #pragma HLS allocation function instances=store_16_ap_fixed_16_5_ limit=1

    #pragma HLS interface m_axi port=DRAM_A1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A4 offset=slave bundle=mem2
    
    #pragma HLS interface m_axi port=DRAM_B1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B4 offset=slave bundle=mem2

    //TOP_A
    load_16_64_ap_fixed_16_5_(DRAM_A1, BRAM_A1);
    load_64_ap_fixed_16_5_(DRAM_A2, BRAM_A2);
    load_16_ap_fixed_16_5_(DRAM_A3, BRAM_A3);
    mmv_ij(BRAM_A1, BRAM_A2, BRAM_A4);
    store_16_ap_fixed_16_5_(BRAM_A4, DRAM_A4);

    //TOP_B
    load_64_16_ap_fixed_16_5_(DRAM_B1, BRAM_B1);
    load_64_ap_fixed_16_5_(DRAM_B2, BRAM_B2);
    load_16_ap_fixed_16_5_(DRAM_B3, BRAM_B3);
    // vmm_with_mmv_ij(BRAM_B1, BRAM_B2, BRAM_B4);
    transpose(BRAM_B1); // Transpose the input matrix A
    mmv_ij(BRAM_B1, BRAM_B2, BRAM_B4); // Call the MMV_IJ function
    store_16_ap_fixed_16_5_(BRAM_B4, DRAM_B4);
}