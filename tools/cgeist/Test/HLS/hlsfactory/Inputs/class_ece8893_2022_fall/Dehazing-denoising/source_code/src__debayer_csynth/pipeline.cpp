#include "pipeline.h"

void img_pipeline(INT_TYPE in[IMG_HEIGHT][IMG_WIDTH], INT_TYPE out[IMG_HEIGHT][IMG_WIDTH][3])
{

#pragma HLS INTERFACE m_axi depth = 384 * 512 port = in bundle = x
#pragma HLS INTERFACE m_axi depth = 384 * 512 * 3 port = out bundle = y

// #pragma HLS INTERFACE s_axilite register port = return

    INT_TYPE vmin[3] = {76, 72, 68};
    #pragma HLS ARRAY_PARTITION variable = vmin complete dim = 1
    INT_TYPE vmax[3] = {169, 170, 169};
    #pragma HLS ARRAY_PARTITION variable = vmax complete dim = 1


    INT_TYPE in_tile[IN_TILE_HEIGHT][IN_TILE_WIDTH] = {0};
// #pragma HLS ARRAY_PARTITION variable = in_tile complete dim = 1

    INT_TYPE debayer_wb_tile[TILE_HEIGHT][TILE_WIDTH][3];
#pragma HLS ARRAY_PARTITION variable = debayer_wb_tile complete dim = 3
// #pragma HLS ARRAY_PARTITION variable = debayer_wb_tile block factor=10 dim = 2

//     INT_TYPE out_tile[TILE_HEIGHT][TILE_WIDTH][3];
// #pragma HLS ARRAY_PARTITION variable = out_tile complete dim = 3

    LONG_INT_TYPE blue_hist[COLOR_INT];
    LONG_INT_TYPE green_hist[COLOR_INT];
    LONG_INT_TYPE red_hist[COLOR_INT];

    for (int i = 0; i < COLOR_INT; i++)
    {
        blue_hist[i] = 0;
        green_hist[i] = 0;
        red_hist[i] = 0;
    }

    for (int i = 0; i < N_TILES_Y; i++)
    {
        for (int j = 0; j < N_TILES_X; j++)
        {
            load_input_tile_from_DRAM(in_tile, in, i, j);

            debayer(in_tile, debayer_wb_tile, i, j, blue_hist, green_hist, red_hist);

            wb_apply_transform_tile(debayer_wb_tile, vmin, vmax);

            tile_dehaze(debayer_wb_tile);

            // Other Image processing blocks
            
            store_output_tile_to_DRAM(debayer_wb_tile, out, i, j);
        }
    }

    // update vmin, vmax
    update_vmin_vmax(blue_hist, green_hist, red_hist, vmin, vmax);
}
