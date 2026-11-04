#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(data_t DRAM_A1[16][64], data_t DRAM_A2[64], data_t DRAM_A3[16], data_t DRAM_A4[16],
    data_t DRAM_B1[64][16], data_t DRAM_B2[64], data_t DRAM_B3[16], data_t DRAM_B4[16]);

#endif // TOP_H