
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

data_t BRAM_1[64];
data_t BRAM_2[64];
data_t BRAM_3[1];
data_t BRAM_4[1];

void load_64_ap_fixed_16_5_(data_t input[64], data_t output[64])
{
    for (int idx0 = 0; idx0 < 64; idx0++) {
        output[idx0] = input[idx0];
    }
}

void load_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

//////////////////////////////////////////
// Begin: DOT_PRODUCT FUNCTION
//////////////////////////////////////////
/*==== DOT_PRODUCT FUNCTION START ====*/
void dot_product(
    ap_fixed<16, 5> input_A[64],
    ap_fixed<16, 5> input_B[64],
    ap_fixed<16, 5> output[1]
)
{
#pragma HLS array_partition variable=input_A cyclic factor=16 dim=1
#pragma HLS array_partition variable=input_B cyclic factor=16 dim=1

output[0] = 0;

for (int i = 0; i < 64; i++) {
#pragma HLS unroll factor=16
    output[0] += input_A[i] * input_B[i];
}
}
/*==== DOT_PRODUCT FUNCTION END ====*/
//////////////////////////////////////////
// END: DOT_PRODUCT FUNCTION
//////////////////////////////////////////


void store_1_ap_fixed_16_5_(data_t input[1], data_t output[1])
{
    for (int idx0 = 0; idx0 < 1; idx0++) {
        output[idx0] = input[idx0];
    }
}

void top(data_t DRAM_1[64], data_t DRAM_2[64], data_t DRAM_3[1], data_t DRAM_4[1])
{
    #pragma HLS interface m_axi port=DRAM_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_4 offset=slave bundle=mem2

    load_64_ap_fixed_16_5_(DRAM_1, BRAM_1);
    load_64_ap_fixed_16_5_(DRAM_2, BRAM_2);
    load_1_ap_fixed_16_5_(DRAM_3, BRAM_3);
    //////////////////////////////////////////
    // Begin: Call to DOT_PRODUCT
//////////////////////////////////////////
dot_product(BRAM_1, BRAM_2, BRAM_4);
//////////////////////////////////////////
// End: Call to DOT_PRODUCT
//////////////////////////////////////////

    store_1_ap_fixed_16_5_(BRAM_4, DRAM_4);
}