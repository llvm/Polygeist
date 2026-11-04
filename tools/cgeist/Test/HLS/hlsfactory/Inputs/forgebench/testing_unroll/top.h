#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[16][8], data_t DRAM_2[8][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32], data_t DRAM_5[32], data_t DRAM_10[32], data_t DRAM_11[1], data_t DRAM_12[1]);

#endif // TOP_H