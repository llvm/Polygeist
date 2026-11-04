#include "pipeline.h"

/* Function to apply a single debayering fiter and generate output */
INT_TYPE applyKernel(INT_TYPE in_3x3[9], INT_TYPE kernel_3x3[9])
{
    INT_TYPE out = (in_3x3[0] * kernel_3x3[0] + in_3x3[1] * kernel_3x3[1] + in_3x3[2] * kernel_3x3[2] +
                    in_3x3[3] * kernel_3x3[3] + in_3x3[4] * kernel_3x3[4] + in_3x3[5] * kernel_3x3[5] +
                    in_3x3[6] * kernel_3x3[6] + in_3x3[7] * kernel_3x3[7] + in_3x3[8] * kernel_3x3[8]) >>
                   2;
    return out;
}

// INT_TYPE debMul(INT_TYPE in_3x3, INT_TYPE kernel_3x3, INT_TYPE factor)
// {
//     INT_TYPE out = (in_3x3 * ((kernel_3x3 >> factor) & 0xF)) >> 2;
//     return out;
// }

void updateHist(INT_TYPE val, LONG_INT_TYPE hist[COLOR_INT])
{
    hist[val]++;
}

void debayer(INT_TYPE in[IN_TILE_HEIGHT][IN_TILE_WIDTH], INT_TYPE out[TILE_HEIGHT][TILE_WIDTH][3], int ti, int tj,
             LONG_INT_TYPE blue_hist[COLOR_INT], LONG_INT_TYPE green_hist[COLOR_INT], LONG_INT_TYPE red_hist[COLOR_INT])
{

    const int height_offset = ti * IN_TILE_HEIGHT;
    const int width_offset = tj * IN_TILE_WIDTH;

    // Combine these???? Use some 2 bits and combine them to single 10 bit value or use 2 bit value only???
    // Combine and SHift to applykernel?? -> Linear Interpolation Kernel??
    /*INT_TYPE hor_K[9] = { 0, 0, 0, 2, 0, 2, 0, 0, 0};
    INT_TYPE ver_K[9] = { 0, 2, 0, 0, 0, 0, 0, 2, 0};
    INT_TYPE cross_K[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
    INT_TYPE diag_K[9] = { 1, 0, 1, 0, 0, 0, 1, 0, 1};
    INT_TYPE static_K[9] = { 0, 0, 0, 0, 4, 0, 0, 0, 0};*/

    // static, hor, ver, diag encoded as 1 type
    // cross, static, static, cross
    // diag, ver, hor, static
    INT16_TYPE blue_K[9] = {0x0001, 0x0200, 0x0001, 0x0020, 0x4000, 0x0020, 0x0001, 0x0200, 0x0001};
    INT16_TYPE green_K[9] = {0x0000, 0x1001, 0x0000, 0x1001, 0x0440, 0x1001, 0x0000, 0x1001, 0x0000};
    INT16_TYPE red_K[9] = {0x1000, 0x0020, 0x1000, 0x0200, 0x0004, 0x0200, 0x1000, 0x0020, 0x1000};

    for (INT_TYPE i = 1; i < IN_TILE_HEIGHT - 1; i++)
    {
        for (INT_TYPE j = 1; j < IN_TILE_WIDTH - 1; j++)
        {

// #pragma HLS pipeline

            INT_TYPE tempRed[9], tempGreen[9], tempBlue[9];
            INT_TYPE factor = (((height_offset + i - 1) & 1) | (((width_offset + j - 1) & 1) << 1)) << 2;

#pragma HLS array_partition variable = tempRed dim = 1 complete
#pragma HLS array_partition variable = tempGreen dim = 1 complete
#pragma HLS array_partition variable = tempBlue dim = 1 complete

            out[i - 1][j - 1][0] = 0;
            out[i - 1][j - 1][1] = 0;
            out[i - 1][j - 1][2] = 0;

            for (INT_TYPE kv = 0; kv < 3; kv++)
            {
                for (INT_TYPE kh = 0; kh < 3; kh++)
                {
                    tempRed[kv * 3 + kh] = (in[i + kv - 1][j + kh - 1] * ((red_K[kv * 3 + kh] >> factor) & 0xF)) >> 2;
                    tempGreen[kv * 3 + kh] = (in[i + kv - 1][j + kh - 1] * ((green_K[kv * 3 + kh] >> factor) & 0xF)) >> 2;
                    tempBlue[kv * 3 + kh] = (in[i + kv - 1][j + kh - 1] * ((blue_K[kv * 3 + kh] >> factor) & 0xF)) >> 2;
                    // tempRed[kv*3 + kh] = debMul(in[i+kv-1][j+kh-1], red_K[kv*3 + kh], factor);
                    // tempGreen[kv*3 + kh] = debMul(in[i+kv-1][j+kh-1], green_K[kv*3 + kh], factor);
                    // tempBlue[kv*3 + kh] = debMul(in[i+kv-1][j+kh-1], blue_K[kv*3 + kh], factor);
                }
            }

            for (int c = 0; c < 9; c++)
            {
                out[i - 1][j - 1][0] += tempRed[c];
                out[i - 1][j - 1][1] += tempGreen[c];
                out[i - 1][j - 1][2] += tempBlue[c];
            }

            updateHist(out[i - 1][j - 1][0], blue_hist);
            updateHist(out[i - 1][j - 1][1], green_hist);
            updateHist(out[i - 1][j - 1][2], red_hist);
        }
    }

    /*for(INT_TYPE i=1; i<IN_TILE_HEIGHT-1; i++){
        for(INT_TYPE j=1; j<IN_TILE_WIDTH-1; j++){
            #pragma HLS pipeline
            //std::cout << "Entry i: " << i << ", j: " <<  j << std::endl;

            INT_TYPE in_3x3[9] = {in[i-1][j-1],in[i-1][j],in[i-1][j+1],
                        in[i][j-1],in[i][j],in[i][j+1],
                        in[i+1][j-1],in[i+1][j],in[i+1][j+1]};

            //Combine conditions to make a 2 bit signal later???
            if(((height_offset + i - 1)%2 == 0) && ((width_offset + j - 1)%2 == 0)){
                out[i-1][j-1][0] = applyKernel(in_3x3, static_K);
                out[i-1][j-1][1] = applyKernel(in_3x3, cross_K);
                out[i-1][j-1][2] = applyKernel(in_3x3, diag_K);
            }
            else if(((height_offset + i - 1)%2 == 0) && ( (width_offset + j - 1)%2 != 0)){
                out[i-1][j-1][0] = applyKernel(in_3x3, hor_K);
                out[i-1][j-1][1] = applyKernel(in_3x3, static_K);
                out[i-1][j-1][2] = applyKernel(in_3x3, ver_K);
            }
            else if(((height_offset + i - 1)%2 != 0) && ( (width_offset + j - 1)%2 == 0)){
                out[i-1][j-1][0] = applyKernel(in_3x3, ver_K);
                out[i-1][j-1][1] = applyKernel(in_3x3, static_K);
                out[i-1][j-1][2] = applyKernel(in_3x3, hor_K);
            }
            else{
                out[i-1][j-1][0] = applyKernel(in_3x3, diag_K);
                out[i-1][j-1][1] = applyKernel(in_3x3, cross_K);
                out[i-1][j-1][2] = applyKernel(in_3x3, static_K);
            }

            updateHist(out[i-1][j-1][0], blue_hist);
            updateHist(out[i-1][j-1][1], green_hist);
            updateHist(out[i-1][j-1][2], red_hist);
            //std::cout << "Exit: i: " << i << ", j: " <<  j << std::endl;
        }
    }*/
}
