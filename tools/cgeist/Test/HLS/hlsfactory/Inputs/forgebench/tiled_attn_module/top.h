#include <ap_fixed.h>
#ifndef TOP_H
#define TOP_H

typedef ap_fixed<16, 5> data_t;

void top( data_t DRAM_A1[8][256], data_t DRAM_A2[256][256], data_t DRAM_A3[256][256], data_t DRAM_A4[256][256],
    data_t DRAM_B1[8][256], data_t DRAM_B2[64][256], data_t DRAM_B3[64][256], data_t DRAM_B4[64][256]);

#endif // TOP_H