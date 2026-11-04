#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_1[256][128], data_t DRAM_2[128][192], data_t DRAM_3[256][192], data_t DRAM_4[256][192]);

#endif // TOP_H