#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(
    data_t DRAM_A1[16][64], data_t DRAM_A2[64][32], data_t DRAM_A3[16][32], data_t DRAM_A4[16][32],
    data_t DRAM_B1[16][64], data_t DRAM_B2[64], data_t DRAM_B3[16], data_t DRAM_B4[16],
    data_t DRAM_C1[64], data_t DRAM_C2[64], data_t DRAM_C3[1], data_t DRAM_C4[1]
);

#endif // TOP_H