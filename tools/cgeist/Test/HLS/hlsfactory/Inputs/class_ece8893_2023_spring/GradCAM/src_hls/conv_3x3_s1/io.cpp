#pragma once
#include "../util.h"
#include "params.hpp"
#include <cassert>

namespace conv_3x3_s1 {

int index_calc(int idx_d, int idx_h, int idx_w, int IN_FM_HEIGHT, int IN_FM_WIDTH)
{
    #pragma HLS inline off
    return idx_d*IN_FM_HEIGHT*IN_FM_WIDTH + idx_h*IN_FM_WIDTH + idx_w;
}

template<int BUF_DEPTH, int BUF_HEIGHT, int BUF_WIDTH, 
int TILE_HEIGHT, int TILE_WIDTH, int PADDING>
void load_fm_tile_block_from_DRAM (
    fm_t in_fm_buf[BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const fm_t in_fm[],
    const int DEPTH_CHECK, const int FM_HEIGHT, const int FM_WIDTH,
    const int ti,
    const int tj,
    const int tk,
    const bool stride_2
)
{

    const int depth_offset = tk * BUF_DEPTH;
    const int height_offset = ti * TILE_HEIGHT * (stride_2 ? 2 : 1);
    const int width_offset  = tj * TILE_WIDTH * (stride_2 ? 2 : 1);

    //static_assert(BUF_HEIGHT == TILE_HEIGHT + 2*PADDING, "BUF_HEIGHT != TILE_HEIGHT + 2*PADDING");
    //static_assert(BUF_WIDTH == TILE_WIDTH + 2*PADDING, "BUF_WIDTH != TILE_WIDTH + 2*PADDING");

    const int P = PADDING;
    const int SCAN_HEIGHT = 2*P + (stride_2 ? 2*TILE_HEIGHT : TILE_HEIGHT);
    const int SCAN_WIDTH = 2*P + (stride_2 ? 2*TILE_WIDTH : TILE_WIDTH);

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < DEPTH_CHECK; c++) // FM and BUF have same depth
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < SCAN_HEIGHT; i++)
        {
            for(int j = 0; j < SCAN_WIDTH; j++)
            {
                #pragma HLS PIPELINE II=1 
                int idx_w = width_offset - P + j;
                int idx_h = height_offset - P + i;
                int idx_d = depth_offset + c;
                //int idx = (idx_w) + (idx_h)*FM_WIDTH + (idx_d)*FM_WIDTH*FM_HEIGHT;
                int idx = conv_3x3_s1::index_calc(idx_d, idx_h, idx_w, FM_HEIGHT, FM_WIDTH);

                if ((idx_h < 0 || idx_h >= FM_HEIGHT) ||
                    (idx_w < 0 || idx_w >= FM_WIDTH))
                {
                    in_fm_buf[c][i][j] = (fm_t) 0;
                }
                else
                {
                    in_fm_buf[c][i][j] = in_fm[idx];
                }
            }
        }
    }
}


void load_layer_params_from_DRAM (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    const wt_t weights[],
    const wt_t bias[],
    const dim_t OUT_FM_DEPTH,
    const dim_t IN_FM_DEPTH,
    const int tk,
    const int tl
)
{

    //assert(IN_FM_DEPTH <= IN_BUF_DEPTH);
    const int kernel_offset  = tk * OUT_BUF_DEPTH;
    const int tl_offset = tl * IN_BUF_DEPTH;

    int DEPTH_CHECK = IN_BUF_DEPTH < IN_FM_DEPTH ? IN_BUF_DEPTH : IN_FM_DEPTH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < DEPTH_CHECK; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < KERNEL_HEIGHT; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < KERNEL_WIDTH; kw++)
	            {
                    #pragma HLS PIPELINE II=1
                    int idx_f = (kernel_offset + f)*IN_FM_DEPTH*KERNEL_HEIGHT*KERNEL_WIDTH;
                    int idx = idx_f + conv_3x3_s1::index_calc(c + tl_offset, kh, kw, 
                                                              KERNEL_HEIGHT, 
                                                              KERNEL_WIDTH);

                    weight_buf[f][c][kh][kw] = weights[idx];
                }
            }
        }
    }

    BIAS:
    if (tl == 0)
        for(int f = 0; f < OUT_BUF_DEPTH; f++)
        {
            #pragma HLS PIPELINE II=1
            bias_buf[f] = bias[kernel_offset + f];
        }

}

template<int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH>
void store_output_tile_to_DRAM (
    fm_t out_fm[],
    const fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const fm_dims_s out_fm_dims,
    const int  ti,
    const int  tj,
    const int  tk,
    const int  relu
)
{

    const int OUT_FM_DEPTH = out_fm_dims.depth;
    const int OUT_FM_HEIGHT = out_fm_dims.height;
    const int OUT_FM_WIDTH = out_fm_dims.width;

    const dim_t depth_offset  = tk * OUT_BUF_DEPTH;
    const dim_t height_offset = ti * OUT_BUF_HEIGHT;
    const dim_t width_offset  = tj * OUT_BUF_WIDTH;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                #pragma HLS PIPELINE II=1
                int idx = conv_3x3_s1::index_calc(depth_offset + f, 
                                                  height_offset + i, 
                                                  width_offset + j, 
                                                  OUT_FM_HEIGHT, 
                                                  OUT_FM_WIDTH);

                fm_t out;
                // ReLU in-place
                if(relu & (out_fm_buf[f][i][j] < (fm_t) 0))
                    out = (fm_t) 0;
                else
                    out = out_fm_buf[f][i][j];

                out_fm[idx] = out;
            }
        }
    }
}

}
