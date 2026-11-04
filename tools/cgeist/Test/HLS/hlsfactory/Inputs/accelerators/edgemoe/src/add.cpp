#include "../include/add.hpp"

void compute_add(patch_blocks_t x, patch_blocks_t y, patch_blocks_t out)
{
    #pragma HLS inline off

    fm_block_t* x_blocks = reinterpret_cast<fm_block_t*>(x);
    fm_block_t* y_blocks = reinterpret_cast<fm_block_t*>(y);
    fm_block_t* out_blocks = reinterpret_cast<fm_block_t*>(out);

    FOR_EACH(i, NUM_PATCHES * NUM_FEATURE_BLOCKS)
    {
        #pragma HLS pipeline
        #pragma HLS dependence variable=x_blocks inter false
        #pragma HLS dependence variable=y_blocks inter false
        #pragma HLS dependence variable=out_blocks inter false

        out_blocks[i] = x_blocks[i] + y_blocks[i];
    }
}
