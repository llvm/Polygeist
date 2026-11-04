#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

top(
    data_t DRAM_A1[96][512], data_t DRAM_A2[512][128], data_t DRAM_A3[96][128], data_t DRAM_A4[96][128],
    data_t DRAM_B1[128][256], data_t DRAM_B2[256][64], data_t DRAM_B3[128][64], data_t DRAM_B4[128][64],
    data_t DRAM_C1[256][128], data_t DRAM_C2[128][192], data_t DRAM_C3[256][192], data_t DRAM_C4[256][192]
);

#endif // TOP_H