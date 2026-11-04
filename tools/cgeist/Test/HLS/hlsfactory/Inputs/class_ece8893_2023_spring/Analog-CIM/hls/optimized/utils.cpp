#include "utils.h"

//--------------------------------------------------------------------------
// Function to load an input tile block from from off-chip DRAM 
// to on-chip BRAM.
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM (
    int in_fm_buf[IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    int in_fm[IN_ROWS][IN_COLS], 
    int ti, 
    int tj 
    )
{
  const int height_offset = ti * TILE_HEIGHT;  
  const int width_offset  = tj * TILE_WIDTH;

  INPUT_BUFFER_HEIGHT:
  for(int i = 0; i < IN_BUF_HEIGHT; i++) {
    INPUT_BUFFER_WIDTH:
    for(int j = 0; j < IN_BUF_WIDTH; j++) {
      in_fm_buf[i][j] = in_fm[height_offset + i][width_offset + j];      
    }
  }
}

//--------------------------------------------------------------------------
// Function to v_ref for analog CIM simulation.
//--------------------------------------------------------------------------
void load_weight_tile_block_from_DRAM (
    f_t weight_buf[WT_BUF_HEIGHT][WT_BUF_WIDTH],
    f_t weights[WT_ROWS][WT_BIN_COLS],
    int tj,
    int wj
    )
{
#pragma HLS inline off

  const int height_offset = tj * WT_BUF_HEIGHT;  
  const int width_offset  = wj * WT_BUF_WIDTH;

  WEIGHT_BUFFER_HEIGHT:
  for(int i = 0; i < WT_BUF_HEIGHT; i++) {
    WEIGHT_BUFFER_WIDTH:
    for(int j = 0; j < WT_BUF_WIDTH; j++) {
      weight_buf[i][j] = weights[height_offset + i][width_offset + j];      
    }
  }
}

//--------------------------------------------------------------------------
// Function to v_ref for analog CIM simulation.
//--------------------------------------------------------------------------
void load_v_ref_from_DRAM (
    f_t v_ref_buf[ADC_LEVELS],
    f_t v_ref[ADC_LEVELS]
    )
{
#pragma HLS inline off

  V_REF_BUFFER:
  for(int i = 0; i < ADC_LEVELS; i++) {
    v_ref_buf[i] = v_ref[i];
  }

}

//--------------------------------------------------------------------------
// Function to cim_args for analog CIM simulation
//--------------------------------------------------------------------------
void load_cim_args_from_DRAM (
    int cim_args_buf[NUM_ARGS],
    int cim_args[NUM_ARGS]
    )
{
#pragma HLS inline off

  CIM_ARGS_BUFFER:
  for(int i = 0; i < NUM_ARGS; i++) {
    cim_args_buf[i] = cim_args[i];
  }

}

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//--------------------------------------------------------------------------
void store_output_tile_block_to_DRAM(
    int out_fm[IN_ROWS][WT_COLS], 
    int out_fm_buf[OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int ti, 
    int wj 
    )
{
  const int height_offset = ti * OUT_BUF_HEIGHT;  
  const int width_offset  = wj * OUT_BUF_WIDTH;

  OUTPUT_BUFFER_HEIGHT:
  for(int i = 0; i < OUT_BUF_HEIGHT; i++) {
    OUTPUT_BUFFER_WIDTH:
    for(int j = 0; j < OUT_BUF_WIDTH; j++) {
      out_fm[height_offset + i][width_offset + j] += out_fm_buf[i][j];      
    }
  }
}
