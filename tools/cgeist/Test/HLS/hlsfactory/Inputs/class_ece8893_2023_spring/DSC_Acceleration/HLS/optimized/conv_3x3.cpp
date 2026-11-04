// ///////////////////////////////////////////////////////////////////////////////
// // Author:      <>
// // Course:      ECE8893 - Parallel Programming for FPGAs
// // Filename:    conv_3x3.cpp
// // Description: Implement a functionally-correct synthesizable 7x7 convolution 
// //              for a single tile block without any optimizations
// ///////////////////////////////////////////////////////////////////////////////
// #include "utils.h"

// void conv_3x3 (
//     fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
//     fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
//     wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
//     wt_t B_buf[OUT_BUF_DEPTH]
// )
// {
// //---------------------------------------------------------------------------
// // Part B: Implement a trivial functionally-correct single-tile convolution.
// //
// //         This should have an overall latency in the order of 22-23 seconds.
// //
// //         If it's worse than trivial, it may be worth fixing this first.
// //         Otherwise, achieving the target latency with a worse-than-trivial
// //         baseline may be difficult!
// //
// // TODO: Your code for Part B goes here. 
// //---------------------------------------------------------------------------

//     FILTER_OUTPUT_DEPTH:
//     for (int f = 0; f < OUT_BUF_DEPTH; f++)
//     {
//         INPUT_IMAGE_HEIGHT:
//         for (int i = 0; i < OUT_BUF_HEIGHT; i++)
//         {
//             INPUT_IMAGE_WIDTH:
//             for (int j = 0; j < OUT_BUF_WIDTH; j++)
//             {
//                 INPUT_IMAGE_DEPTH:
//                 fm_t sum = 0;
//                 for (int c = 0; c < IN_BUF_DEPTH; c++)
//                 {
//                     KERNEL_FILTER_HEIGHT:
//                     for (int kh = 0; kh < KERNEL_HEIGHT; kh++)
//                     {
//                         KERNEL_FILTER_WIDTH:
//                         for (int kw = 0; kw < KERNEL_WIDTH; kw++)       
//                         {
//                             // Multiple and Accumulate (MAC)
//                             int ki = (i * STRIDE) + kh;
//                             int kj = (j * STRIDE) + kw;
//                             sum += X_buf[c][ki][kj] * W_buf[f][c][kh][kw];
//                         }
//                     }
//                 }
//                 Y_buf[f][i][j] = sum + B_buf[f];//Bias
//             }
//         }
//     }

// }
