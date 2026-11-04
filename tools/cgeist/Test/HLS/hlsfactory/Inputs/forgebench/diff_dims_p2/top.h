#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[128][256], data_t DRAM_2[256][64], data_t DRAM_3[128][64], data_t DRAM_4[128][64]);

#endif // TOP_H