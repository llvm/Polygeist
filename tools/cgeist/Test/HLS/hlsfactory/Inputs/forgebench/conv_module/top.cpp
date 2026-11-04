#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
#include "top.h"
using namespace std;

data_t FM_buffer_1[MAX_C][MAX_H][MAX_W];
data_t weight_buffer[MAX_C][MAX_C][KSIZE][KSIZE];
data_t bias_buffer[MAX_C];
data_t FM_buffer_2[MAX_C][MAX_H][MAX_W];

void conv_kernel_3x3(
    data_t inData[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE],
    data_t weight[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE],
    data_t outData[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W],
    int outTileH, int outTileW,
    int localInH, int localInW,
    int stride
)
{   
    #pragma HLS inline off
    #pragma HLS array_partition variable=inData type=cyclic  factor=8  dim=1
    #pragma HLS array_partition variable=weight type=cyclic  factor=64  dim=1
    #pragma HLS array_partition variable=weight type=cyclic  factor=8  dim=2
    #pragma HLS array_partition variable=outData type=cyclic  factor=64  dim=1
    // Only loop over the valid output tile region
    for (int oh = 0; oh < outTileH; oh++) {
        for (int ow = 0; ow < outTileW; ow++) {
            // 3×3 kernel loops
            for (int kh = 0; kh < KSIZE; kh++) {
                for (int kw = 0; kw < KSIZE; kw++) {
                    // Input channel loop (now second innermost)
                    int in_r = oh * stride + kh;
                    int in_c = ow * stride + kw;
                    for (int ic = 0; ic < BLOCK_IN_CH; ic++) {
                        #pragma HLS unroll factor = 8
                        
                        data_t in_val = inData[ic][in_r][in_c];

                        // Output channel loop (now the innermost)
                        for (int oc = 0; oc < BLOCK_OUT_CH; oc++) {
                        #pragma HLS unroll factor = 64
                            outData[oc][oh][ow] 
                                += in_val * weight[oc][ic][kh][kw];
                        }
                    }
                }
            }
        }
    }
}


void conv_via_tiling_3x3(
    int in_ch,
    int out_ch,
    int H, int W,  // input spatial dimensions
    data_t input[MAX_C][MAX_H][MAX_W],
    data_t weight[MAX_C][MAX_C][KSIZE][KSIZE],
    data_t bias[MAX_C],
    data_t output[MAX_C][MAX_H][MAX_W],
    int stride,
    int pad
)
{
    #pragma HLS inline off
    #pragma HLS allocation function instances= conv_via_tiling_3x3 limit=1
    int out_h = out_dim(H, pad, stride, KSIZE);
    int out_w = out_dim(W, pad, stride, KSIZE);

    // (A) Initialize the full 3‑D output with bias.
    #pragma HLS array_partition variable=bias type=cyclic  factor=64  dim=1
    for (int ow = 0; ow < out_w; ow++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int oc = 0; oc < out_ch; oc++) {
            #pragma HLS unroll factor = 64
                output[oc][oh][ow] = bias[oc];
            }
        }
    }


    // Tiling parameters.
    int tile_rows = ceil_div(out_h, BLOCK_OUT_H);
    int tile_cols = ceil_div(out_w, BLOCK_OUT_W);
    int out_ch_tiles = ceil_div(out_ch, BLOCK_OUT_CH);
    int in_ch_tiles  = ceil_div(in_ch, BLOCK_IN_CH);

    // (B) Loop over spatial tiles.
    for (int tr = 0; tr < tile_rows; tr++) {
        for (int tc = 0; tc < tile_cols; tc++) {
            int out_row0 = tr * BLOCK_OUT_H;
            int out_col0 = tc * BLOCK_OUT_W;
            int tileH = min(BLOCK_OUT_H, out_h - out_row0);
            int tileW = min(BLOCK_OUT_W, out_w - out_col0);

            // Compute local input patch size for this tile.
            int localInH = (tileH - 1) * stride + KSIZE;
            int localInW = (tileW - 1) * stride + KSIZE;

            // (C) Loop over output-channel tiles.
            for (int oct = 0; oct < out_ch_tiles; oct++) {
                int oc_offset = oct * BLOCK_OUT_CH;
                int cur_out_ch = (oc_offset + BLOCK_OUT_CH <= out_ch) ? BLOCK_OUT_CH : (out_ch - oc_offset);

                // Copy current partial sums from global output into a local buffer.
                data_t localOut[BLOCK_OUT_CH][BLOCK_OUT_H][BLOCK_OUT_W];
                for (int block_out_ch_i = 0; block_out_ch_i < BLOCK_OUT_CH; block_out_ch_i++)
                {
                    for (int block_out_h_i = 0; block_out_h_i < BLOCK_OUT_H; block_out_h_i++)
                    {
                        for (int block_out_w_i = 0; block_out_w_i < BLOCK_OUT_W; block_out_w_i++)
                        {
                            localOut[block_out_ch_i][block_out_h_i][block_out_w_i] = (data_t) 0;
                        }
                    }
                }
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int r_local = 0; r_local < tileH; r_local++) {
                        for (int c_local = 0; c_local < tileW; c_local++) {
                            localOut[oc2][r_local][c_local] =
                                output[oc_offset + oc2][out_row0 + r_local][out_col0 + c_local];
                        }
                    }
                }

                // (D) Loop over input-channel tiles.
                for (int ict = 0; ict < in_ch_tiles; ict++) {
                    int ic_offset = ict * BLOCK_IN_CH;
                    int cur_in_ch = (ic_offset + BLOCK_IN_CH <= in_ch) ? BLOCK_IN_CH : (in_ch - ic_offset);

                    // Build localW for the current slice.
                    data_t localW[BLOCK_OUT_CH][BLOCK_IN_CH][KSIZE][KSIZE];

                    for (int block_out_ch_j = 0; block_out_ch_j < BLOCK_OUT_CH; block_out_ch_j++)
                    {
                        for (int block_in_ch_j = 0; block_in_ch_j < BLOCK_IN_CH; block_in_ch_j++)
                        {
                            for (int k_i = 0; k_i < KSIZE; k_i++)
                            {
                                for (int k_j = 0; k_j < KSIZE; k_j++)
                                {
                                    localW[block_out_ch_j][block_in_ch_j][k_i][k_j] = (data_t) 0;
                                }
                            }
                        }
                    }
                    for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                        for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                            for (int kr = 0; kr < KSIZE; kr++) {
                                for (int kc = 0; kc < KSIZE; kc++) {
                                    localW[oc2][ic2][kr][kc] =
                                        weight[oc_offset + oc2][ic_offset + ic2][kr][kc];
                                }
                            }
                        }
                    }

                    // Build localIn for the current input-channel tile.
                    data_t localIn[BLOCK_IN_CH][MAX_LOCAL_SIZE][MAX_LOCAL_SIZE];
                    
                    for (int block_in_ch_k = 0; block_in_ch_k < BLOCK_IN_CH; block_in_ch_k++)
                    {
                        for (int block_in_h_k = 0; block_in_h_k < MAX_LOCAL_SIZE; block_in_h_k++)
                        {
                            for (int block_in_w_k = 0; block_in_w_k < MAX_LOCAL_SIZE; block_in_w_k++)
                            {
                                localIn[block_in_ch_k][block_in_h_k][block_in_w_k] = (data_t) 0;
                            }
                        }
                    }
                    for (int ic2 = 0; ic2 < cur_in_ch; ic2++) {
                        for (int rr = 0; rr < localInH; rr++) {
                            for (int cc = 0; cc < localInW; cc++) {
                                int global_r = out_row0 * stride - pad + rr;
                                int global_c = out_col0 * stride - pad + cc;
                                if (global_r >= 0 && global_r < H && global_c >= 0 && global_c < W) {
                                    localIn[ic2][rr][cc] = input[ic_offset + ic2][global_r][global_c];
                                }
                            }
                        }
                    }

                    // (E) Call conv_kernel for this tile.
                    conv_kernel_3x3(localIn, localW, localOut, tileH, tileW, localInH, localInW, stride);
                } // end input-channel tile loop

                // (F) Write the updated localOut back to the global 3‑D output.
                for (int oc2 = 0; oc2 < cur_out_ch; oc2++) {
                    for (int r = 0; r < tileH; r++) {
                        for (int c = 0; c < tileW; c++) {
                            output[oc_offset + oc2][out_row0 + r][out_col0 + c] = localOut[oc2][r][c];
                        }
                    }
                }
            } // end output-channel tile loop
        } // end spatial tile loop (tc)
    } // end spatial tile loop (tr)
}


void load_feature_map(data_t input_dram[MAX_C][MAX_H][MAX_W], data_t input_buffer[MAX_C][MAX_H][MAX_W], int C, int H, int W)
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

void store_feature_map(data_t output_buffer[MAX_C][MAX_H][MAX_W], data_t output_dram[MAX_C][MAX_H][MAX_W], int C, int H, int W)
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

void load_weights(data_t weight_dram[MAX_C][MAX_C][KSIZE][KSIZE], data_t weight_buffer[MAX_C][MAX_C][KSIZE][KSIZE], int C_OUT, int C_IN, int K)
{
    #pragma HLS inline off
    for (int cout = 0; cout < C_OUT; cout++)
    {
        for (int cin = 0; cin < C_IN; cin++)
        {
            for (int k1 = 0; k1 < K; k1++)
            {
                for (int k2 = 0; k2 < K; k2++)
                {
                    weight_buffer[cout][cin][k1][k2] = weight_dram[cout][cin][k1][k2];
                }
                
            }
        }
    }
}

void load_bias(data_t bias_dram[MAX_C], data_t bias_buffer[MAX_C], int C)
{
    #pragma HLS inline off
    for (int c = 0; c < C; c++)
    {
        bias_buffer[c] = bias_dram[c];
    }
}
// conv: output channel, input channel, kernel, kernel, stride, padding
// basic unified logic input: 64, 7, 7 conv: 64, 64, 3, 3 output 64, 7, 7

// A: input: 64, 14, 14 conv: 64, 64, 3, 3, 1, 1 output: 64, 14, 14
// B: input: 128, 7, 7 conv: 128, 128, 3, 3, 1, 1 output: 128, 7, 7
// C: input: 128, 14, 14 conv: 128, 128, 3, 3, 1, 1 output: 128, 14, 14

// D: input: 64, 14, 14 conv: 64, 128, 3, 3, 1, 1 output: 128, 14, 14
// E: input: 128, 7, 7 conv: 128, 64, 3, 3, 1, 1 output: 64, 7, 7
// F: input: 128, 14, 14 conv: 128, 64, 3, 3, 1, 1 output 64, 14, 14

// G: input: 64, 14, 14 conv: 64, 64, 3, 3, 1, 0 output: 64, 12, 12
// H: input: 64, 14, 14 conv: 64, 64, 3, 3, 2, 1 output: 64, 7, 7
// I: input: 64, 28, 28 conv: 64, 64, 3, 3, 2, 0 output: 64, 13, 13


void top_A(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 64, 14, 14);
    load_weights(conv_weight, weight_buffer, 64, 64, 3);
    load_bias(conv_bias, bias_buffer, 64);
    conv_via_tiling_3x3(64, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output, 64, 14, 14);
}

void top_B(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 128, 7, 7);
    load_weights(conv_weight, weight_buffer, 128, 128, 3);
    load_bias(conv_bias, bias_buffer, 128);
    conv_via_tiling_3x3(128, 128, 7, 7, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output, 128, 7, 7);
}

void top_C(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 128, 14, 14);
    load_weights(conv_weight, weight_buffer, 128, 128, 3);
    load_bias(conv_bias, bias_buffer, 128);
    conv_via_tiling_3x3(128, 128, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output, 128, 14, 14);
}

void top_D(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 64, 14, 14);
    load_weights(conv_weight, weight_buffer, 128, 64, 3);
    load_bias(conv_bias, bias_buffer, 128);
    conv_via_tiling_3x3(64, 128, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output, 128, 14, 14);
}

void top_E(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 128, 7, 7);
    load_weights(conv_weight, weight_buffer, 64, 128, 3);
    load_bias(conv_bias, bias_buffer, 64);
    conv_via_tiling_3x3(128, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output, 64, 14, 14);
}

void top_F(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 128, 14, 14);
    load_weights(conv_weight, weight_buffer, 64, 128, 3);
    load_bias(conv_bias, bias_buffer, 64);
    conv_via_tiling_3x3(128, 64, 7, 7, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output, 64, 7, 7);
}




void top_G(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 64, 14, 14);
    load_weights(conv_weight, weight_buffer, 64, 64, 3);
    load_bias(conv_bias, bias_buffer, 64);
    conv_via_tiling_3x3(64, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 0);
    store_feature_map(FM_buffer_2, output, 64, 12, 12);
}

void top_H(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 64, 14, 14);
    load_weights(conv_weight, weight_buffer, 64, 64, 3);
    load_bias(conv_bias, bias_buffer, 64);
    conv_via_tiling_3x3(64, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 2, 1);
    store_feature_map(FM_buffer_2, output, 64, 7, 7);
}

void top_I(data_t input [MAX_C][MAX_H][MAX_W], 
            data_t conv_weight[MAX_C][MAX_C][KSIZE][KSIZE], 
            data_t conv_bias[MAX_C],
            data_t output[MAX_C][MAX_H][MAX_W]
        )
{
    #pragma HLS inline off
    load_feature_map(input, FM_buffer_1, 64, 28, 28);
    load_weights(conv_weight, weight_buffer, 64, 64, 3);
    load_bias(conv_bias, bias_buffer, 64);
    conv_via_tiling_3x3(64, 64, 28, 28, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 2, 0);
    store_feature_map(FM_buffer_2, output, 64, 13, 13);
}



void top(
    data_t input_A [MAX_C][MAX_H][MAX_W], 
    data_t conv_weight_A[MAX_C][MAX_C][KSIZE][KSIZE], 
    data_t conv_bias_A[MAX_C],
    data_t output_A[MAX_C][MAX_H][MAX_W],

    data_t input_B [MAX_C][MAX_H][MAX_W], 
    data_t conv_weight_B[MAX_C][MAX_C][KSIZE][KSIZE], 
    data_t conv_bias_B[MAX_C],
    data_t output_B[MAX_C][MAX_H][MAX_W],

    data_t input_C [MAX_C][MAX_H][MAX_W], 
    data_t conv_weight_C[MAX_C][MAX_C][KSIZE][KSIZE], 
    data_t conv_bias_C[MAX_C],
    data_t output_C[MAX_C][MAX_H][MAX_W]

    // data_t input_D [MAX_C][MAX_H][MAX_W], 
    // data_t conv_weight_D[MAX_C][MAX_C][KSIZE][KSIZE], 
    // data_t conv_bias_D[MAX_C],
    // data_t output_D[MAX_C][MAX_H][MAX_W],

    // data_t input_E [MAX_C][MAX_H][MAX_W], 
    // data_t conv_weight_E[MAX_C][MAX_C][KSIZE][KSIZE], 
    // data_t conv_bias_E[MAX_C],
    // data_t output_E[MAX_C][MAX_H][MAX_W],

    // data_t input_F [MAX_C][MAX_H][MAX_W], 
    // data_t conv_weight_F[MAX_C][MAX_C][KSIZE][KSIZE], 
    // data_t conv_bias_F[MAX_C],
    // data_t output_F[MAX_C][MAX_H][MAX_W],

    // data_t input_G [MAX_C][MAX_H][MAX_W], 
    // data_t conv_weight_G[MAX_C][MAX_C][KSIZE][KSIZE], 
    // data_t conv_bias_G[MAX_C],
    // data_t output_G[MAX_C][MAX_H][MAX_W],

    // data_t input_H [MAX_C][MAX_H][MAX_W], 
    // data_t conv_weight_H[MAX_C][MAX_C][KSIZE][KSIZE], 
    // data_t conv_bias_H[MAX_C],
    // data_t output_H[MAX_C][MAX_H][MAX_W],

    // data_t input_I [MAX_C][MAX_H][MAX_W], 
    // data_t conv_weight_I[MAX_C][MAX_C][KSIZE][KSIZE], 
    // data_t conv_bias_I[MAX_C],
    // data_t output_I[MAX_C][MAX_H][MAX_W]


)
{
    #pragma HLS interface m_axi port=input_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=conv_weight_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=conv_bias_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_A offset=slave bundle=mem2

    #pragma HLS interface m_axi port=input_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=conv_weight_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=conv_bias_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_B offset=slave bundle=mem2

    #pragma HLS interface m_axi port=input_C offset=slave bundle=mem1
    #pragma HLS interface m_axi port=conv_weight_C offset=slave bundle=mem1
    #pragma HLS interface m_axi port=conv_bias_C offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_C offset=slave bundle=mem2

    // #pragma HLS interface m_axi port=input_D offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_weight_D offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_bias_D offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=output_D offset=slave bundle=mem2

    // #pragma HLS interface m_axi port=input_E offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_weight_E offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_bias_E offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=output_E offset=slave bundle=mem2

    // #pragma HLS interface m_axi port=input_F offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_weight_F offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_bias_F offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=output_F offset=slave bundle=mem2

    // #pragma HLS interface m_axi port=input_G offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_weight_G offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_bias_G offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=output_G offset=slave bundle=mem2

    // #pragma HLS interface m_axi port=input_H offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_weight_H offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_bias_H offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=output_H offset=slave bundle=mem2

    // #pragma HLS interface m_axi port=input_I offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_weight_I offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=conv_bias_I offset=slave bundle=mem1
    // #pragma HLS interface m_axi port=output_I offset=slave bundle=mem2



    #pragma HLS allocation function instances= load_feature_map limit=1
    #pragma HLS allocation function instances= load_weights limit=1
    #pragma HLS allocation function instances= load_bias limit=1
    #pragma HLS allocation function instances= conv_via_tiling_3x3 limit=1
    #pragma HLS allocation function instances= store_feature_map limit=1

    
    //top_A
    load_feature_map(input_A, FM_buffer_1, 64, 14, 14);
    load_weights(conv_weight_A, weight_buffer, 64, 64, 3);
    load_bias(conv_bias_A, bias_buffer, 64);
    conv_via_tiling_3x3(64, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output_A, 64, 14, 14);

    //top_B
    load_feature_map(input_B, FM_buffer_1, 128, 7, 7);
    load_weights(conv_weight_B, weight_buffer, 128, 128, 3);
    load_bias(conv_bias_B, bias_buffer, 128);
    conv_via_tiling_3x3(128, 128, 7, 7, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output_B, 128, 7, 7);

    //top_C
    load_feature_map(input_C, FM_buffer_1, 128, 14, 14);
    load_weights(conv_weight_C, weight_buffer, 128, 128, 3);
    load_bias(conv_bias_C, bias_buffer, 128);
    conv_via_tiling_3x3(128, 128, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    store_feature_map(FM_buffer_2, output_C, 128, 14, 14);

    // //top_D
    // load_feature_map(input_D, FM_buffer_1, 64, 14, 14);
    // load_weights(conv_weight_D, weight_buffer, 128, 64, 3);
    // load_bias(conv_bias_D, bias_buffer, 128);
    // conv_via_tiling_3x3(64, 128, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    // store_feature_map(FM_buffer_2, output_D, 128, 14, 14);

    // //top_E
    // load_feature_map(input_E, FM_buffer_1, 128, 7, 7);
    // load_weights(conv_weight_E, weight_buffer, 64, 128, 3);
    // load_bias(conv_bias_E, bias_buffer, 64);
    // conv_via_tiling_3x3(128, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    // store_feature_map(FM_buffer_2, output_E, 64, 14, 14);

    // //top_F
    // load_feature_map(input_F, FM_buffer_1, 128, 14, 14);
    // load_weights(conv_weight_F, weight_buffer, 64, 128, 3);
    // load_bias(conv_bias_F, bias_buffer, 64);
    // conv_via_tiling_3x3(128, 64, 7, 7, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 1);
    // store_feature_map(FM_buffer_2, output_F, 64, 7, 7);

    // //top_G
    // load_feature_map(input_G, FM_buffer_1, 64, 14, 14);
    // load_weights(conv_weight_G, weight_buffer, 64, 64, 3);
    // load_bias(conv_bias_G, bias_buffer, 64);
    // conv_via_tiling_3x3(64, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 1, 0);
    // store_feature_map(FM_buffer_2, output_G, 64, 12, 12);

    // //top_H
    // load_feature_map(input_H, FM_buffer_1, 64, 14, 14);
    // load_weights(conv_weight_H, weight_buffer, 64, 64, 3);
    // load_bias(conv_bias_H, bias_buffer, 64);
    // conv_via_tiling_3x3(64, 64, 14, 14, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 2, 1);
    // store_feature_map(FM_buffer_2, output_H, 64, 7, 7);

    // //top_I
    // load_feature_map(input_I, FM_buffer_1, 64, 28, 28);
    // load_weights(conv_weight_I, weight_buffer, 64, 64, 3);
    // load_bias(conv_bias_I, bias_buffer, 64);
    // conv_via_tiling_3x3(64, 64, 28, 28, FM_buffer_1, weight_buffer, bias_buffer, FM_buffer_2, 2, 0);
    // store_feature_map(FM_buffer_2, output_I, 64, 13, 13);

    // top_A(input_A, conv_weight_A, conv_bias_A, output_A);
    // top_B(input_B, conv_weight_B, conv_bias_B, output_B);
    // top_C(input_C, conv_weight_C, conv_bias_C, output_C);
    // top_D(input_D, conv_weight_D, conv_bias_D, output_D);
    // top_E(input_E, conv_weight_E, conv_bias_E, output_E);
    // top_F(input_F, conv_weight_F, conv_bias_F, output_F);
    // top_G(input_G, conv_weight_G, conv_bias_G, output_G);
    // top_H(input_H, conv_weight_H, conv_bias_H, output_H);
    // top_I(input_I, conv_weight_I, conv_bias_I, output_I);

}
