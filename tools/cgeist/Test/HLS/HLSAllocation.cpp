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

#define C 64
#define H 28
#define W 28

data_t FM_buffer_1[C][H][W];
data_t FM_buffer_2[C][H][W];
data_t FM_buffer_3[C][H][W];
data_t FM_buffer_4[C][H][W];


void load_feature_map(data_t input_dram[C][H][W], data_t input_buffer[C][H][W])
{
    #pragma HLS inline off
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                input_buffer[c][h][w] = input_dram[c][h][w];
            }
        }
    }
}

void store_feature_map(data_t output_buffer[C][H][W], data_t output_dram[C][H][W])
{
    #pragma HLS inline off
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                output_dram[c][h][w] = output_buffer[c][h][w];
            }
        }
    }
}

void compute_exp(data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = hls::exp(input[i][j][k]);
            }
        }
    }

}

void negative(data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = -input[i][j][k];
            }
        }
    }
}

void compute_div(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = input_1[i][j][k]/input_2[i][j][k];
            }
        }
    }
}

void compute_add(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = input_1[i][j][k] + input_2[i][j][k];
            }
        }
    }
}

void compute_mul(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                output[i][j][k] = input_1[i][j][k] * input_2[i][j][k];
            }
        }
    }
}

void set_value(data_t bram[C][H][W], data_t value)
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                bram[i][j][k] = value;
            }
        }
    }
}


void select(data_t input_1[C][H][W], data_t input_2[C][H][W], data_t output[C][H][W],  data_t threshold)
{
    #pragma HLS inline off
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                if (input_1[i][j][k] >= threshold) {
                    output[i][j][k] = input_1[i][j][k];
                } else {
                    output[i][j][k] = input_2[i][j][k];
                }
            }
        }
    }
}

void sigmoid (data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1);
    negative(FM_buffer_1, FM_buffer_2);
    compute_exp(FM_buffer_2, FM_buffer_1);
    set_value(FM_buffer_3, (data_t) 1);
    compute_add(FM_buffer_3, FM_buffer_1, FM_buffer_2);
    compute_div(FM_buffer_3, FM_buffer_2, FM_buffer_1);
    store_feature_map(FM_buffer_1, output);
}

void tanh (data_t input[C][H][W], data_t output[C][H][W])
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1);
    compute_exp(FM_buffer_1, FM_buffer_2); //exp(x)
    negative(FM_buffer_1, FM_buffer_3);
    compute_exp(FM_buffer_3, FM_buffer_1); //exp(-x)
    compute_add(FM_buffer_2, FM_buffer_1, FM_buffer_3);
    negative(FM_buffer_1, FM_buffer_4);
    compute_add(FM_buffer_2, FM_buffer_4, FM_buffer_1);
    compute_div(FM_buffer_1, FM_buffer_3, FM_buffer_2);
    store_feature_map(FM_buffer_2, output);
}

void elu(data_t input[C][H][W], data_t output[C][H][W], data_t alpha)
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1);
    compute_exp(FM_buffer_1, FM_buffer_2);
    set_value(FM_buffer_3, (data_t) -1);
    compute_add(FM_buffer_2, FM_buffer_3, FM_buffer_4);
    set_value(FM_buffer_3, alpha);
    compute_mul(FM_buffer_3, FM_buffer_4, FM_buffer_2);
    select(FM_buffer_1, FM_buffer_2, FM_buffer_3, (data_t) 0);
    store_feature_map(FM_buffer_3, output);
}




void top(
    data_t input_sigmoid[C][H][W],
    data_t output_sigmoid[C][H][W], 
    data_t input_tanh[C][H][W],
    data_t output_tanh[C][H][W],
    data_t input_elu[C][H][W],
    data_t output_elu[C][H][W]
)
{
    #pragma HLS interface m_axi port=input_sigmoid offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_sigmoid offset=slave bundle=mem2

    #pragma HLS interface m_axi port=input_tanh offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_tanh offset=slave bundle=mem2

    #pragma HLS interface m_axi port=input_elu offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_elu offset=slave bundle=mem2


    #pragma HLS allocation function instances= load_feature_map limit=1
    #pragma HLS allocation function instances= store_feature_map limit=1
    #pragma HLS allocation function instances= compute_exp limit=1
    #pragma HLS allocation function instances= negative limit=1
    #pragma HLS allocation function instances= compute_div limit=1
    #pragma HLS allocation function instances= compute_add limit=1
    #pragma HLS allocation function instances= compute_mul limit=1
    #pragma HLS allocation function instances= set_value limit=1
    #pragma HLS allocation function instances= select limit=1
   
    //sigmoid
    load_feature_map(input_sigmoid, FM_buffer_1);
    negative(FM_buffer_1, FM_buffer_2);
    compute_exp(FM_buffer_2, FM_buffer_1);
    set_value(FM_buffer_3, (data_t) 1);
    compute_add(FM_buffer_3, FM_buffer_1, FM_buffer_2);
    compute_div(FM_buffer_3, FM_buffer_2, FM_buffer_1);
    store_feature_map(FM_buffer_1, output_sigmoid);

    //tanh
    load_feature_map(input_tanh, FM_buffer_1);
    compute_exp(FM_buffer_1, FM_buffer_2); //exp(x)
    negative(FM_buffer_1, FM_buffer_3);
    compute_exp(FM_buffer_3, FM_buffer_1); //exp(-x)
    compute_add(FM_buffer_2, FM_buffer_1, FM_buffer_3);
    negative(FM_buffer_1, FM_buffer_4);
    compute_add(FM_buffer_2, FM_buffer_4, FM_buffer_1);
    compute_div(FM_buffer_1, FM_buffer_3, FM_buffer_2);
    store_feature_map(FM_buffer_2, output_tanh);

    //elu
    load_feature_map(input_elu, FM_buffer_1);
    compute_exp(FM_buffer_1, FM_buffer_2);
    set_value(FM_buffer_3, (data_t) -1);
    compute_add(FM_buffer_2, FM_buffer_3, FM_buffer_4);
    set_value(FM_buffer_3, (data_t) 0.5);
    compute_mul(FM_buffer_3, FM_buffer_4, FM_buffer_2);
    select(FM_buffer_1, FM_buffer_2, FM_buffer_3, (data_t) 0);
    store_feature_map(FM_buffer_3, output_elu);
    // sigmoid(input_sigmoid, output_sigmoid);
    // tanh(input_tanh, output_tanh);
    // elu(input_elu, output_elu, (data_t) 0.5);
}


// Vitis ap_fixed/hls_math code: the pragmas are checked without code
// generation. (With the earlier stand-in headers cgeist could not generate
// code for hls::exp on ap_fixed, math.exp got the struct operand; with the
// exactclass headers and --scalar-wrapper-abi it does.)
// RUN: cgeist %s --function='*' -S --hls-annotate --hls-pragma-check-only %stdinclude %vitis_include --scalar-wrapper-abi -o /dev/null 2>&1 | FileCheck %s --check-prefix=HLS
// HLS-NOT: error
// HLS: HLSAllocation.cpp:27:5: remark: '#pragma HLS inline off' -> func.func @load_feature_map
// HLS: HLSAllocation.cpp:42:5: remark: '#pragma HLS inline off' -> func.func @store_feature_map
// HLS: HLSAllocation.cpp:57:5: remark: '#pragma HLS inline off' -> func.func @compute_exp
// HLS: HLSAllocation.cpp:73:5: remark: '#pragma HLS inline off' -> func.func @negative
// HLS: HLSAllocation.cpp:88:5: remark: '#pragma HLS inline off' -> func.func @compute_div
// HLS: HLSAllocation.cpp:103:5: remark: '#pragma HLS inline off' -> func.func @compute_add
// HLS: HLSAllocation.cpp:118:5: remark: '#pragma HLS inline off' -> func.func @compute_mul
// HLS: HLSAllocation.cpp:133:5: remark: '#pragma HLS inline off' -> func.func @set_value
// HLS: HLSAllocation.cpp:149:5: remark: '#pragma HLS inline off' -> func.func @select
// HLS: HLSAllocation.cpp:168:5: remark: '#pragma HLS inline off' -> func.func @sigmoid
// HLS: HLSAllocation.cpp:180:5: remark: '#pragma HLS inline off' -> func.func @tanh
// HLS: HLSAllocation.cpp:194:5: remark: '#pragma HLS inline off' -> func.func @elu
// HLS: HLSAllocation.cpp:217:5: remark: '#pragma HLS interface m_axi port=input_sigmoid offset=slave bundle=mem1' -> argument #0 (variable 'input_sigmoid') of @top
// HLS: HLSAllocation.cpp:218:5: remark: '#pragma HLS interface m_axi port=output_sigmoid offset=slave bundle=mem2' -> argument #1 (variable 'output_sigmoid') of @top
// HLS: HLSAllocation.cpp:220:5: remark: '#pragma HLS interface m_axi port=input_tanh offset=slave bundle=mem1' -> argument #2 (variable 'input_tanh') of @top
// HLS: HLSAllocation.cpp:221:5: remark: '#pragma HLS interface m_axi port=output_tanh offset=slave bundle=mem2' -> argument #3 (variable 'output_tanh') of @top
// HLS: HLSAllocation.cpp:223:5: remark: '#pragma HLS interface m_axi port=input_elu offset=slave bundle=mem1' -> argument #4 (variable 'input_elu') of @top
// HLS: HLSAllocation.cpp:224:5: remark: '#pragma HLS interface m_axi port=output_elu offset=slave bundle=mem2' -> argument #5 (variable 'output_elu') of @top
// HLS: HLSAllocation.cpp:227:5: remark: '#pragma HLS allocation function instances= load_feature_map limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:228:5: remark: '#pragma HLS allocation function instances= store_feature_map limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:229:5: remark: '#pragma HLS allocation function instances= compute_exp limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:230:5: remark: '#pragma HLS allocation function instances= negative limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:231:5: remark: '#pragma HLS allocation function instances= compute_div limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:232:5: remark: '#pragma HLS allocation function instances= compute_add limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:233:5: remark: '#pragma HLS allocation function instances= compute_mul limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:234:5: remark: '#pragma HLS allocation function instances= set_value limit=1' -> func.func @top
// HLS: HLSAllocation.cpp:235:5: remark: '#pragma HLS allocation function instances= select limit=1' -> func.func @top
// HLS-NOT: remark:
