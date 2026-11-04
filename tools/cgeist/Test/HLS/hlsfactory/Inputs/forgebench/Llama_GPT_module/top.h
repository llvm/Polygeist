#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>

#ifndef TOP_H
#define TOP_H


typedef ap_fixed<16, 5> data_t;

#define SEQ_LENGTH 8
#define DIM 32

void top(data_t DRAM_attn_input_gpt[SEQ_LENGTH][DIM], data_t DRAM_weights_q_gpt[DIM][DIM], data_t DRAM_weights_k_gpt[DIM][DIM], data_t DRAM_weights_v_gpt[DIM][DIM], data_t DRAM_layer_norm_weights_1_gpt[2][DIM], data_t DRAM_FF_weights_1_gpt[4 * DIM][DIM], data_t DRAM_FF_weights_2_gpt[DIM][4 * DIM], data_t DRAM_layer_norm_weights_2_gpt[2][DIM], data_t DRAM_output_gpt[SEQ_LENGTH][DIM],
        data_t DRAM_attn_input_llama[SEQ_LENGTH][DIM], data_t DRAM_weights_q_llama[DIM][DIM], data_t DRAM_weights_k_llama[DIM][DIM], data_t DRAM_weights_v_llama[DIM][DIM], data_t DRAM_rms_norm_weights_1_llama[2][DIM], data_t DRAM_FF_weights_1_llama[4 * DIM][DIM], data_t DRAM_FF_weights_2_llama[DIM][4 * DIM], data_t DRAM_rms_norm_weights_2_llama[2][DIM], data_t DRAM_output_llama[SEQ_LENGTH][DIM]);

#endif // TOP_H