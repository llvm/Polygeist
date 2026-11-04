#include "params.h"
#include <stdio.h>
void histogram_hls(
    unsigned char data[DATA_SIZE / KNOB_NUM_WORK_GROUPS],
    unsigned int histogram[KNOB_HIST_SIZE],
    unsigned long offset) {
    // using this to correctly put in precompiled defs in pragmas
    unsigned int hist1[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 2
    unsigned int hist2[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 3
    unsigned int hist3[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 4
    unsigned int hist4[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 5
    unsigned int hist5[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 6
    unsigned int hist6[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 7
    unsigned int hist7[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 8
    unsigned int hist8[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 9
    unsigned int hist9[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 10
    unsigned int hist10[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 11
    unsigned int hist11[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 12
    unsigned int hist12[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 13
    unsigned int hist13[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 14
    unsigned int hist14[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 15
    unsigned int hist15[KNOB_HIST_SIZE];
#if KNOB_NUM_HIST >= 16
    unsigned int hist16[KNOB_HIST_SIZE];
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
    // begin loop for initializing the arrays
init_loop:
    for (int i = 0; i < KNOB_HIST_SIZE; i++) {
        hist1[i] = 0;

#if KNOB_NUM_HIST >= 2
        hist2[i] = 0;

#if KNOB_NUM_HIST >= 3
        hist3[i] = 0;

#if KNOB_NUM_HIST >= 4
        hist4[i] = 0;

#if KNOB_NUM_HIST >= 5
        hist5[i] = 0;

#if KNOB_NUM_HIST >= 6
        hist6[i] = 0;

#if KNOB_NUM_HIST >= 7
        hist7[i] = 0;

#if KNOB_NUM_HIST >= 8
        hist8[i] = 0;

#if KNOB_NUM_HIST >= 9
        hist9[i] = 0;

#if KNOB_NUM_HIST >= 10
        hist10[i] = 0;

#if KNOB_NUM_HIST >= 11
        hist11[i] = 0;

#if KNOB_NUM_HIST >= 12
        hist12[i] = 0;

#if KNOB_NUM_HIST >= 13
        hist13[i] = 0;

#if KNOB_NUM_HIST >= 14
        hist14[i] = 0;

#if KNOB_NUM_HIST >= 15
        hist15[i] = 0;

#if KNOB_NUM_HIST >= 16
        hist16[i] = 0;

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
    }

    unsigned long count;
    // assigning values to each histogram specified max-KNOB_NUM_HIST+1

hist_loop:
    for (count = 0; count < TRIPCNT; count += KNOB_NUM_HIST) {

        hist1[data[count + offset]]++;

#if KNOB_NUM_HIST >= 2
        hist2[data[count + offset + 1]]++;

#if KNOB_NUM_HIST >= 3
        hist3[data[count + offset + 2]]++;

#if KNOB_NUM_HIST >= 4
        hist4[data[count + offset + 3]]++;

#if KNOB_NUM_HIST >= 5
        hist5[data[count + offset + 4]]++;

#if KNOB_NUM_HIST >= 6
        hist6[data[count + offset + 5]]++;

#if KNOB_NUM_HIST >= 7
        hist7[data[count + offset + 6]]++;

#if KNOB_NUM_HIST >= 8
        hist8[data[count + offset + 7]]++;

#if KNOB_NUM_HIST >= 9
        hist9[data[count + offset + 8]]++;

#if KNOB_NUM_HIST >= 10
        hist10[data[count + offset + 9]]++;

#if KNOB_NUM_HIST >= 11
        hist11[data[count + 10]]++;

#if KNOB_NUM_HIST >= 12
        hist12[data[count + offset + 11]]++;

#if KNOB_NUM_HIST >= 13
        hist13[data[count + offset + 12]]++;

#if KNOB_NUM_HIST >= 14
        hist14[data[count + offset + 13]]++;

#if KNOB_NUM_HIST >= 15
        hist15[data[count + offset + 14]]++;

#if KNOB_NUM_HIST >= 16
        hist16[data[count + offset + 15]]++;

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
    }

#if LEFTOVER_LOOP >= 1
    hist1[data[count + offset]]++;

#if LEFTOVER_LOOP >= 2
    hist2[data[count + offset + 1]]++;

#if LEFTOVER_LOOP >= 3
    hist3[data[count + offset + 2]]++;

#if LEFTOVER_LOOP >= 4
    hist4[data[count + offset + 3]]++;

#if LEFTOVER_LOOP >= 5
    hist5[data[count + offset + 4]]++;

#if LEFTOVER_LOOP >= 6
    hist6[data[count + offset + 5]]++;

#if LEFTOVER_LOOP >= 7
    hist7[data[count + offset + 6]]++;

#if LEFTOVER_LOOP >= 8
    hist8[data[count + offset + 7]]++;

#if LEFTOVER_LOOP >= 9
    hist9[data[count + offset + 8]]++;

#if LEFTOVER_LOOP >= 10
    hist10[data[count + offset + 9]]++;

#if LEFTOVER_LOOP >= 11
    hist11[data[count + offset + 10]]++;

#if LEFTOVER_LOOP >= 12
    hist12[data[count + offset + 11]]++;

#if LEFTOVER_LOOP >= 13
    hist13[data[count + offset + 12]]++;

#if LEFTOVER_LOOP >= 14
    hist14[data[count + offset + 13]]++;

#if LEFTOVER_LOOP >= 15
    hist15[data[count + offset + 14]]++;

#if LEFTOVER_LOOP >= 16
    hist16[data[count + offset + 15]]++;

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif

    // Accumulate the local buffers (if using several)

accum_loop:
    for (int i = 0; i < KNOB_HIST_SIZE; i++) {

        histogram[i] += hist1[i] +

#if KNOB_NUM_HIST >= 2
                        hist2[i] +

#if KNOB_NUM_HIST >= 3
                        hist3[i] +

#if KNOB_NUM_HIST >= 4
                        hist4[i] +

#if KNOB_NUM_HIST >= 5
                        hist5[i] +

#if KNOB_NUM_HIST >= 6
                        hist6[i] +

#if KNOB_NUM_HIST >= 7
                        hist7[i] +

#if KNOB_NUM_HIST >= 8
                        hist8[i] +

#if KNOB_NUM_HIST >= 9
                        hist9[i] +

#if KNOB_NUM_HIST >= 10
                        hist10[i] +

#if KNOB_NUM_HIST >= 11
                        hist11[i] +

#if KNOB_NUM_HIST >= 12
                        hist12[i] +

#if KNOB_NUM_HIST >= 13
                        hist13[i] +

#if KNOB_NUM_HIST >= 14
                        hist14[i] +

#if KNOB_NUM_HIST >= 15
                        hist15[i] +

#if KNOB_NUM_HIST >= 16
                        hist16[i] +

#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
                        0;
    }
}
