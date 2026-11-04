#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[96][512], data_t DRAM_2[512][128], data_t DRAM_3[96][128], data_t DRAM_4[96][128]);

#endif // TOP_H