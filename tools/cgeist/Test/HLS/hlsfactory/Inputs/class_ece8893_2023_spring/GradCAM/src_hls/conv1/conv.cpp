#pragma once
#include "../util.h"

#include "params.hpp"

void conv_small (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    const wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t B_buf[OUT_BUF_DEPTH]
)
{
#pragma HLS INLINE off
    const int S = STRIDE;

    #pragma HLS ARRAY_PARTITION variable=W_buf type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Y_buf type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_buf type=complete dim=1

IN_FEAT:  for (int id = 0; id < IN_BUF_DEPTH; id++) {
    IN_ROW:   for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
        IN_COL:   for (int kw = 0; kw < KERNEL_WIDTH; kw++)  {
            OUT_COL:  for (int ow = 0; ow < OUT_BUF_WIDTH; ow++) {
                OUT_ROW:  for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++) {
                            #pragma HLS PIPELINE II=1
                    OUT_FEAT: for (int of = 0; of < OUT_BUF_DEPTH; of++) { // 16
                            #pragma HLS unroll

                            int i = S*oh + kh;
                            int j = S*ow + kw;

                            fm_t x = Y_buf[of][oh][ow];

                            if (id == 0 && kh == 0 && kw == 0)
                            {
                                x = B_buf[of] + X_buf[id][i][j] * W_buf[of][id][kh][kw];
                            }
                            else
                            {
                                x += X_buf[id][i][j] * W_buf[of][id][kh][kw];
                            }
                            Y_buf[of][oh][ow] = x;
                        }}}}}}
}
