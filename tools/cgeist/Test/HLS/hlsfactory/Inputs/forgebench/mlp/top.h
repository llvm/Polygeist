#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[256][64], data_t DRAM_2[64][128], data_t DRAM_3[256][128], data_t DRAM_4[128][16], data_t DRAM_5[256][16], data_t DRAM_6[256][16]);

#endif // TOP_H