#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_attn_input[8][64], data_t DRAM_weights_q[64][64], data_t DRAM_weights_k[64][64], data_t DRAM_weights_v[64][64], data_t DRAM_output[8][64]);

#endif // TOP_H