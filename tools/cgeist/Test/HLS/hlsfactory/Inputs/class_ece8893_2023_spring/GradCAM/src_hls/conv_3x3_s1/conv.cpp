#pragma once
#include "../util.h"
#include "params.hpp"

void conv_small(
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    const fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    const wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    const wt_t B_buf[OUT_BUF_DEPTH],
    const int  IN_FM_DEPTH,
    const bool stride_2,
    const bool add_to_output,
    const bool add_bias
)
{
#pragma HLS INLINE off

    int S = stride_2 ? 2 : 1;

    #pragma HLS ARRAY_PARTITION variable=Y_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=W_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=1

    if (add_bias) {
    B_D: for (int ow = 0; ow < OUT_BUF_WIDTH; ow++) {
        B_H: for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++) {
                #pragma HLS PIPELINE II=1
            B_W: for (int of = 0; of < OUT_BUF_DEPTH; of++) {
                    #pragma HLS UNROLL
                    fm_t x = Y_buf[of][oh][ow];
                    if (add_to_output)
                        x += B_buf[of];
                    else
                        x = B_buf[of];
                    Y_buf[of][oh][ow] = x;
                }
            }
        }
    }

IN_FEAT:  for (int id = 0; id < IN_BUF_DEPTH; id++) {
    IN_ROW:   for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
        IN_COL:   for (int kw = 0; kw < KERNEL_WIDTH; kw++)  {
            OUT_COL:  for (int ow = 0; ow < OUT_BUF_WIDTH; ow++) {
                OUT_ROW:  for (int oh = 0; oh < OUT_BUF_HEIGHT; oh++) {
                    #pragma HLS PIPELINE II=1
                    OUT_FEAT: for (int of = 0; of < OUT_BUF_DEPTH; of++) {
                                #pragma HLS UNROLL
                                fm_t x = Y_buf[of][oh][ow];
                                int i = S*oh + kh;
                                int j = S*ow + kw;
                                x += X_buf[id][i][j] * W_buf[of][id][kh][kw];
                                Y_buf[of][oh][ow] = x;
                        }}}}}}
}
