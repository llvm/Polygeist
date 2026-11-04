#include "utils.h"

void tiled_cim_conv(
  int input2d[IN_ROWS][IN_COLS],
  f_t weight2d_cond[WT_ROWS][WT_BIN_COLS],
  f_t v_ref[ADC_LEVELS],
  int output[IN_ROWS][WT_COLS],
  int cim_args[NUM_ARGS]
) 
{

#pragma HLS INTERFACE m_axi depth=1 port=input2d offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi depth=1 port=weight2d_cond offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi depth=1 port=v_ref offset=slave bundle=mem2
#pragma HLS INTERFACE m_axi depth=1 port=output offset=slave bundle=mem2
#pragma HLS INTERFACE m_axi depth=1 port=cim_args offset=slave bundle=mem3

#pragma HLS INTERFACE s_axilite register	port=return

  // on-chip buffers
  int input_buf[IN_BUF_HEIGHT][IN_BUF_WIDTH];
  f_t weight2d_cond_buf[WT_BUF_HEIGHT][WT_BUF_WIDTH];
  f_t v_ref_buf[ADC_LEVELS];
  int cim_args_buf[NUM_ARGS];
  int output_buf[OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

  // load v_ref
  load_v_ref_from_DRAM(v_ref_buf, v_ref);

  // load cim_args
  load_cim_args_from_DRAM(cim_args_buf, cim_args);

  // process each tile
  TILE_ROW:
  for (int ti = 0; ti < N_TILE_ROWS; ti++) {
    TILE_COL:
    for (int tj = 0; tj < N_TILE_COLS; tj++) {
      //std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
      //std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;

      // load input tile block 
      load_input_tile_block_from_DRAM(input_buf, input2d, ti, tj); 

      TILE_WEIGHT:
      for (int wj = 0; wj < N_OUT_COLS; wj++) {
        // load weight tile block
        load_weight_tile_block_from_DRAM(weight2d_cond_buf, weight2d_cond, tj, wj);

        // compute output tile block
        cim_conv(input_buf, weight2d_cond_buf, v_ref_buf, cim_args_buf, output_buf);

        // store output tile block
        store_output_tile_block_to_DRAM(output, output_buf, ti, wj);
      }
    }
  }
}
