#include "conv.h"

//--------------------------------------------------------------------------
// Function Declarations
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM (
    int in_fm_buf[IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    int in_fm[IN_ROWS][IN_COLS], 
    int  ti, 
    int  tj 
    );

void load_weight_tile_block_from_DRAM (
    f_t weight_buf[WT_BUF_HEIGHT][WT_BUF_WIDTH],
    f_t weights[WT_ROWS][WT_BIN_COLS],
    int  tj,
    int  wj
    );

void load_v_ref_from_DRAM (
    f_t v_ref_buf[ADC_LEVELS],
    f_t v_ref[ADC_LEVELS]
    );

void load_cim_args_from_DRAM (
    int cim_args_buf[NUM_ARGS],
    int cim_args[NUM_ARGS]
    );

void cim_conv(
    int input_buf[IN_BUF_HEIGHT][IN_BUF_WIDTH],
    f_t weight2d_cond_buf[WT_BUF_HEIGHT][WT_BUF_WIDTH],
    f_t v_ref_buf[ADC_LEVELS],
    int cim_args_buf[NUM_ARGS],
    int output_buf[OUT_BUF_HEIGHT][OUT_BUF_WIDTH]
    );

void store_output_tile_block_to_DRAM (
    int out_fm[IN_ROWS][WT_COLS], 
    int out_fm_buf[OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  wj
    );
