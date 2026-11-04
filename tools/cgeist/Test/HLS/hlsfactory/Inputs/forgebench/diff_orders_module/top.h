#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top(
    data_t DRAM_A1[64][128], data_t DRAM_A2[128][64], data_t DRAM_A3[64][64], data_t DRAM_A4[64][64],
    data_t DRAM_B1[32][128], data_t DRAM_B2[128][32], data_t DRAM_B3[32][32], data_t DRAM_B4[32][32],
    data_t DRAM_C1[64][128], data_t DRAM_C2[128][128], data_t DRAM_C3[64][128], data_t DRAM_C4[64][128]
);

#endif // TOP_H