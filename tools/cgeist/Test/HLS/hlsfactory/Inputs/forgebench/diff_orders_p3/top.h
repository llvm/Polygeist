#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[64][128], data_t DRAM_2[128][128], data_t DRAM_3[64][128], data_t DRAM_4[64][128]);

#endif // TOP_H