#pragma once
#include "../util.h"

#include "params.hpp"

template<int TILE_DEPTH, int TILE_HEIGHT, int TILE_WIDTH, int PADDING, int FM_DEPTH, int FM_HEIGHT, int FM_WIDTH>
void load_fm_tile_block_from_DRAM (
    fm_t in_fm_buf[TILE_DEPTH][TILE_HEIGHT + 2*PADDING][TILE_WIDTH + 2*PADDING],
    const fm_t in_fm[FM_DEPTH][FM_HEIGHT][FM_WIDTH],
    const int ti,
    const int tj,
    const int tk
)
{
    const int depth_offset = tk * TILE_DEPTH;
    const int height_offset = ti * TILE_HEIGHT;
    const int width_offset  = tj * TILE_WIDTH;

    const int BUF_DEPTH = TILE_DEPTH;
    const int BUF_HEIGHT = TILE_HEIGHT + 2*PADDING;
    const int BUF_WIDTH = TILE_WIDTH + 2*PADDING;

    const int P = PADDING;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < BUF_DEPTH; c++) // FM and BUF have same depth
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < BUF_HEIGHT; i++)
        {
            for(int j = 0; j < BUF_WIDTH; j++)
            {
                int idx_d = depth_offset + c;
                int idx_h = height_offset + i - P;
                int idx_w = width_offset + j - P;
                if ((idx_h < 0 || idx_h >= FM_HEIGHT) ||
                    (idx_w < 0 || idx_w >= FM_WIDTH))
                {
                    in_fm_buf[c][i][j] = (fm_t) 0;
                }
                else
                {
                    in_fm_buf[c][i][j] = in_fm[idx_d][idx_h][idx_w];
                }
            }
        }
    }
}


void load_layer_params_from_DRAM (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t bias_buf[OUT_BUF_DEPTH],
    const wt_t weights[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t bias[OUT_BUF_DEPTH],
    const int tk
)
{
    const int kernel_offset  = tk * OUT_BUF_DEPTH;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < KERNEL_HEIGHT; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < KERNEL_WIDTH; kw++)
	            {
                    weight_buf[f][c][kh][kw] = weights[kernel_offset + f][c][kh][kw];
                }
            }
        }
    }

    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }

}

template<int OUT_BUF_DEPTH, int OUT_BUF_HEIGHT, int OUT_BUF_WIDTH,
int OUT_FM_DEPTH, int OUT_FM_HEIGHT, int OUT_FM_WIDTH>
void store_output_tile_to_DRAM (
    fm_t out_fm[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH],
    const fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const int  ti,
    const int  tj,
    const int  tk,
    const bool relu
)
{
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
                int idx_d = depth_offset + f;
                int idx_h = height_offset + i;
                int idx_w = width_offset + j;

                fm_t out;
                // ReLU in-place
                if(relu & (out_fm_buf[f][i][j] < (fm_t) 0))
                {
                    out = (fm_t) 0;
                }
                else
                {
                    out = out_fm_buf[f][i][j];
                }

                out_fm[idx_d][idx_h][idx_w] = out;
            }
        }
    }
}
