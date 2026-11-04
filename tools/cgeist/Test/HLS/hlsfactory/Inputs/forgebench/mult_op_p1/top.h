#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[16][64], data_t DRAM_2[64][32], data_t DRAM_3[16][32], data_t DRAM_4[16][32]);

#endif // TOP_H