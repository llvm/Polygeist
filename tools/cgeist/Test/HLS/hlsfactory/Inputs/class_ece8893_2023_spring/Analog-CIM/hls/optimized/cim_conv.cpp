#include "utils.h"

void cim_conv(
  int input_buf[IN_BUF_HEIGHT][IN_BUF_WIDTH],
  f_t weight2d_cond_buf[WT_BUF_HEIGHT][WT_BUF_WIDTH],
  f_t v_ref_buf[ADC_LEVELS],
  int cim_args_buf[NUM_ARGS],
  int output_buf[OUT_BUF_HEIGHT][OUT_BUF_WIDTH]
) 
{

#pragma HLS array_partition variable=weight2d_cond_buf dim=1 complete
#pragma HLS array_partition variable=weight2d_cond_buf dim=2 type=cyclic factor=8
  for (int i = 0; i < IN_BUF_HEIGHT; i++) {
    for (int j = 0; j < OUT_BUF_WIDTH; j++) {
      output_buf[i][j] = 0;

      for (int m = 0; m < INPUT_PRECISION; m++) { // LSB to MSB
        int input_mask = 1 << m;

#pragma HLS pipeline
        for (int k = 0; k < WEIGHT_PRECISION; k++) { // MSB to LSB
          int weight_mask = (1 << (WEIGHT_PRECISION-k-1));

          // calculate the equivalent conductance of this group of rows
          f_t equiv_cond = 0;
          for (int r = 0; r < PARALLEL_ROWS; r++) {  
            int w_col = j*WEIGHT_PRECISION + k;
            int input_bit = (input_buf[i][r] & input_mask) >> m;
            
            equiv_cond += input_bit*weight2d_cond_buf[r][w_col];
          }
  
          // calculate the output voltage using a voltage divider
          f_t temp  = 1 + cim_args_buf[RES_DIV_POS] * equiv_cond;
          f_t v_out = (f_t)cim_args_buf[VDD_POS]/temp;
  
          // sense with the ADC
          int adc_out = ADC_LEVELS;
          for (int l=0; l < ADC_LEVELS; l++) {
            if (v_out >= v_ref_buf[l]) {
                adc_out = l;
                break;
            }
          }
    
          // add to output
          output_buf[i][j] += adc_out * weight_mask * input_mask;
        }
      }
    }
  }
}
