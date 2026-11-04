#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_image_input[64][14][14], data_t DRAM_conv_weight[64][64][3][3], data_t DRAM_conv_bias[64], data_t DRAM_image_output[64][14][14]);

#endif // TOP_H