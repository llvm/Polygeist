#pragma once

#include <cassert>
#include "../util.h"

namespace maxpool 
{
 //#include "params.hpp"
 //#include "conv.cpp"
 //#include "io.cpp"
 //#include "max.cpp"
//

int index_calc(int idx_d, int idx_h, int idx_w, int IN_FM_HEIGHT, int IN_FM_WIDTH)
{
    #pragma HLS inline off
    return idx_d*IN_FM_HEIGHT*IN_FM_WIDTH + idx_h*IN_FM_WIDTH + idx_w;
}

template<int BUF_DEPTH, int BUF_HEIGHT, int BUF_WIDTH, 
int TILE_HEIGHT, int TILE_WIDTH, int PADDING, int FM_HEIGHT, int FM_WIDTH>
void load_fm_tile_block_from_DRAM (
    fm_t in_fm_buf[BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const fm_t in_fm[],
    const int ti,
    const int tj,
    const int tk
)
{

    const int depth_offset = tk * BUF_DEPTH;
    const int height_offset = ti * TILE_HEIGHT;
    const int width_offset  = tj * TILE_WIDTH;
    const int P = PADDING;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < BUF_DEPTH; c++) // FM and BUF have same depth
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < BUF_HEIGHT; i++)
        {
            for(int j = 0; j < BUF_WIDTH; j++)
            {
                #pragma HLS PIPELINE II=1 
                int idx_w = width_offset - P + j;
                int idx_h = height_offset - P + i;
                int idx_d = depth_offset + c;
                int idx = maxpool::index_calc(idx_d, idx_h, idx_w, FM_HEIGHT, FM_WIDTH);

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

template<int OD, int OH, int OW, int ID, int IH, int IW, 
        int KH, int KW, int ST>
void maxpool2d_core(
    fm_t output[OD][OH][OW],
    fm_t input[ID][IH][IW]
)
{
    #pragma HLS inline off
MP_FEAT:
    for (int c = 0; c < OD; c++) {
    MP_ROW:
        for (int h = 0; h < OH; h++) {
        MP_COL:
            for (int w = 0; w < OW; w++) 
            {
                fm_t max_val = -1e30;
                MP_KER_ROW:
                for (int i = 0; i < KH; i++) 
                {
                    MP_KER_COL:
                    for (int j = 0; j < KW; j++) 
                    {
                        int row_idx = (ST * h) + i;
                        int col_idx = (ST * w) + j;
                        assert(row_idx < IH);
                        assert(col_idx < IW);
                        fm_t val = input[c][row_idx][col_idx];
                        if (val > max_val) 
                        {
                            max_val = val;
                        }
                    }
                }
                output[c][h][w] = max_val;
            }
        }
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
                int idx = maxpool::index_calc(depth_offset + f, 
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

void maxpool2d(
    fm_t output[],
    const fm_t input[]
)
{
    #pragma HLS inline off
    #pragma HLS INTERFACE m_axi depth=1  port=input   bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=output bundle=fm_out
    #pragma HLS INTERFACE s_axilite register	port=return

    const int IN_FM_DEPTH = 64;
    const int IN_FM_HEIGHT = 112;
    const int IN_FM_WIDTH = 112;

    const int OUT_FM_DEPTH = 64;
    const int OUT_FM_HEIGHT = 56;
    const int OUT_FM_WIDTH = 56;

    const int P = 1;
    const int KERNEL_HEIGHT = 3;
    const int KERNEL_WIDTH = 3;
    const int TILE_DEPTH = 64;
    const int TILE_HEIGHT = 14;
    const int TILE_WIDTH=14;
    const int N_TILES = (
        (IN_FM_HEIGHT / TILE_HEIGHT) 
        * (IN_FM_WIDTH / TILE_WIDTH) 
        * (IN_FM_DEPTH / TILE_DEPTH));


    const int IN_BUF_HEIGHT = TILE_HEIGHT + 2*P;
    const int IN_BUF_WIDTH = TILE_WIDTH + 2*P;
    const int OUT_BUF_HEIGHT = TILE_HEIGHT/2;
    const int OUT_BUF_WIDTH = TILE_WIDTH/2;

    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH / TILE_WIDTH;

    static fm_t in_buf[TILE_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    static fm_t out_buf[TILE_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {

            // Load to in_buf
            maxpool::load_fm_tile_block_from_DRAM
                <TILE_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH, 
                TILE_HEIGHT, TILE_WIDTH, P, IN_FM_HEIGHT, IN_FM_WIDTH>
                (in_buf, input, ti, tj, 0);

            maxpool::maxpool2d_core<TILE_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, 
                TILE_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH, 
                KERNEL_HEIGHT, KERNEL_WIDTH, 2>
                (out_buf, in_buf);

            maxpool::store_output_tile_to_DRAM
                <TILE_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH>
                (output, out_buf, {OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH}, 
                 ti, tj, 0, 0);
        }

    }
}

}
