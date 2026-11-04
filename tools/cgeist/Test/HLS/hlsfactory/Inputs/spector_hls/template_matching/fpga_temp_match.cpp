#include "fpga_temp_matching.h"

buffer row_buf = {};
window win_buf = {};

void SAD_MATCH(axis_t *INPUT, axis_t *OUTPUT) {
    unsigned char templ[tmpsize] = {0};
#pragma HLS INTERFACE axis depth = 50 port = INPUT
#pragma HLS INTERFACE axis depth = 50 port = OUTPUT
#pragma HLS INTERFACE s_axilite port = return bundle = CTRL

    PRAGMA_HLS(
        HLS array_partition variable = win_buf.win block factor = tmpdim dim =
            0)
    PRAGMA_HLS(HLS array_partition variable = templ block factor = tmpsize)

#pragma HLS dependence variable = row_buf.buf intra true
    //#pragma HLS array_partition variable=row_buf.buf block factor = 10 dim=1
    axis_t cur;
    int cur_data;
    int out[size];
#pragma HLS dependence variable = out intra RAW true
    int i = 0, j = 0, k = 0, o = 0, m = 0, n = 0, l = 0;

    // Pull next pixel into buffer
    for (i = 0; i < size; i++) {
        PRAGMA_HLS(HLS UNROLL FACTOR = UNROLL_FACTOR)
        cur = *INPUT;
        INPUT++;
        cur_data = cur.data;
        row_buf.buf[tmpdim - 1][k] = cur_data;

        // shift window
        for (m = 0; m < tmpdim - 1; m++) {
            PRAGMA_HLS(HLS UNROLL FACTOR = UNROLL_LOOP1)
            for (n = 0; n < tmpdim; n++) {
                win_buf.win[n][m] = win_buf.win[n][m + 1];
            }
        }

        // pull column from buffer into window
        for (l = 0; l < tmpdim; l++) {
            PRAGMA_HLS(HLS UNROLL FACTOR = UNROLL_LOOP2)
            win_buf.win[l][tmpdim - 1] = row_buf.buf[l][k];
        }

        // SAD (Sum of Absolute Differences)
        int y, z, sad = 0;
        int absl = 0;
        for (y = 0; y < tmpdim; y++) {
            PRAGMA_HLS(HLS UNROLL FACTOR = UNROLL_LOOP3)
            for (z = 0; z < tmpdim; z++) {
                absl = win_buf.win[y][z] - templ[z + tmpdim * y] > 0
                           ? win_buf.win[y][z] - templ[z + tmpdim * y]
                           : win_buf.win[y][z] - templ[z + tmpdim * y] * -1;
                sad += absl;
            }
        }

        out[i] = (sad < thre ? 1 : 0);

        // if the buffer row is filled, shift buffer row up by 1
        k++;
        if (k == indim) {
            k = 0;
            for (j = 0; j < tmpdim - 1; j++) {
                PRAGMA_HLS(HLS UNROLL FACTOR = UNROLL_LOOP4)
                for (o = 0; o < indim; o++) {
                    row_buf.buf[j][o] = row_buf.buf[j + 1][o];
                }
            }
        }

        cur.last = 0;
        cur.data = out[i];
        if (i == size - 1) {
            cur.last = 1;
        } else
            cur.last = 0;
        *OUTPUT++ = cur;
    }
}
