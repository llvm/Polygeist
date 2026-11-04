#include "pipeline.h"

void compare_and_swap_unit(INT_TYPE &a, INT_TYPE &b)
{
    if (a < b)
    {
        a = a ^ b;
        b = b ^ a;
        a = a ^ b;
    }
}

void mHMF_7(INT_TYPE input[7], INT_TYPE &median)
{   
// #pragma HLS dataflow
    compare_and_swap_unit(input[4], input[0]); // Cycle 1
    compare_and_swap_unit(input[5], input[1]);
    compare_and_swap_unit(input[6], input[2]);

    compare_and_swap_unit(input[2], input[0]); // Cycle 2
    compare_and_swap_unit(input[3], input[1]);
    compare_and_swap_unit(input[6], input[4]);

    compare_and_swap_unit(input[4], input[2]); // Cycle 3
    compare_and_swap_unit(input[5], input[3]);
    compare_and_swap_unit(input[1], input[0]);

    compare_and_swap_unit(input[5], input[4]); // Cycle 4
    compare_and_swap_unit(input[3], input[2]);

    compare_and_swap_unit(input[6], input[3]); // Cycle 5
    compare_and_swap_unit(input[4], input[1]);

    compare_and_swap_unit(input[4], input[3]); // Cycle 6

    median = input[3]; // Median
}

void mHMF_13(INT_TYPE input[13], INT_TYPE &median)
{
    //median = 0;
    INT_TYPE vals_tmp[7];
// #pragma HLS array_partition variable=vals_tmp complete dim=1
    INT_TYPE m1;
    INT_TYPE m2;

    /*for (unsigned int i = 0; i < 7; i++)
    {
        vals_tmp[i] = 0;
    }*/

    for (unsigned int i = 0; i < 7; i++)
        vals_tmp[i] = input[i];
    mHMF_7(vals_tmp, m1);

    for (unsigned int j = 6; j < 13; j++)
        vals_tmp[j - 6] = input[j];
    mHMF_7(vals_tmp, m2);

    median = (m1 + m2) / 2;
}

void mHMF_49(INT_TYPE input[49], INT_TYPE &median)
{
    //median = 0;
    INT_TYPE vals_tmp[7];
// #pragma HLS array_partition variable=vals_tmp complete dim=1
    INT_TYPE median_tmp[7];
// #pragma HLS array_partition variable=median_tmp complete dim=1
    INT_TYPE m;

    /*for (unsigned int i = 0; i < 7; i++)
    {
        vals_tmp[i] = 0;
        median_tmp[i] = 0;
    }*/

    for (unsigned int i = 0; i < 7; i++)
    {
        for (unsigned int j = 0; j < 7; j++)
        {
            vals_tmp[i] = input[7 * i + j];
        }
        mHMF_7(vals_tmp, m);
        median_tmp[i] = m;
    }
    mHMF_7(median_tmp, m);
    median = m;
}

void calculate_median(INT_TYPE square_window[49], INT_TYPE diag_window[13], INT_TYPE cross_window[13], INT_TYPE &median)
{
    INT_TYPE median_vals[3];
    INT_TYPE m;

    /*for (unsigned int i = 0; i < 3; i++)
    {
        median_vals[i] = 0;
    }*/

    // Calculate median of square window : median_vals[0]
    mHMF_49(square_window, m);
    median_vals[0] = m;

    // Calculate median of diag window : median_vals[1]
    mHMF_13(diag_window, m);
    median_vals[1] = m;

    // Calculate median of cross window : median_vals[2]
    mHMF_13(cross_window, m);
    median_vals[2] = m;

    // Calculate median of medians
    compare_and_swap_unit(median_vals[0], median_vals[1]);
    compare_and_swap_unit(median_vals[1], median_vals[2]);
    compare_and_swap_unit(median_vals[0], median_vals[1]);

    median = median_vals[1];
}

void tile_median(INT_TYPE Color_Tile[TILE_HEIGHT][TILE_WIDTH], INT_TYPE Median_Tile[TILE_HEIGHT][TILE_WIDTH])
{
    for (int i = 0; i < TILE_HEIGHT - 6; i++)
    {
        for (int j = 0; j < TILE_WIDTH - 6; j++)
        {
            INT_TYPE square_window[49] = {0};
            INT_TYPE diag_window[13] = {0};
            INT_TYPE cross_window[13] = {0};

            // Square Window: 49
            for (int k = 0; k < 7; k++)
                for (int l = 0; l < 7; l++)
                {
                    square_window[7 * k + l] = Color_Tile[i + k][j + l];
                }

            // Cross Window: 13
            for (int c = 0; c < 7; c++)
            {
                cross_window[c] = Color_Tile[i + c][j + 3];
            }
            unsigned int index = 7;
            for (int c = 0; c < 7; c++)
            {
                if (c == 3)
                    continue;
                cross_window[index] = Color_Tile[i + 3][j + c];
                index++;
            }

            // Diag Window: 13
            for (int d = 0; d < 7; d++)
            {
                diag_window[d] = Color_Tile[i + d][j + d];
            }
            index = 7;
            for (int d = 0; d < 7; d++)
            {
                if (d == 3)
                    continue;
                diag_window[index] = Color_Tile[i + 6 - d][j + d];
                index++;
            }

            INT_TYPE median = 0;

            calculate_median(square_window, diag_window, cross_window, median);

            Median_Tile[i + 3][j + 3] = median;
        }
    }
}

// Dehaze an input tile (22x22) to output tile (16x16) (remaining pixels are ignored when writing back to DRAM)
void tile_dehaze(INT_TYPE image_tile[TILE_HEIGHT][TILE_WIDTH][3])
{
    // MIN-RGB Tile
    INT_TYPE Min_RGB_Tile[TILE_HEIGHT][TILE_WIDTH];
    // Median Tile
    INT_TYPE Median_Tile[TILE_HEIGHT][TILE_WIDTH];

    // Initialize to 0; avoid garbage values in HLS
    /*for (int i = 0; i < TILE_HEIGHT; i++)
    {
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            Min_RGB_Tile[i][j] = 0;
            Median_Tile[i][j] = 0;
        }
    }*/

    // Find the Min_RGB for the given tile
    for (int i = 0; i < TILE_HEIGHT; i++)
    {
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            INT_TYPE pixel_0 = image_tile[i][j][0];
            INT_TYPE pixel_1 = image_tile[i][j][1];
            INT_TYPE pixel_2 = image_tile[i][j][2];

            // pixel_0 = image[i][j][0];
            // pixel_1 = image[i][j][1];
            // pixel_2 = image[i][j][2];
            
            // Sort
            compare_and_swap_unit(pixel_0, pixel_1);
            compare_and_swap_unit(pixel_1, pixel_2);
            compare_and_swap_unit(pixel_0, pixel_1);
            
            Min_RGB_Tile[i][j] = pixel_2;
        }
    }

    // Input - Min_RGB_Tile
    // Output - Median_Tile after running the mHMF
    tile_median(Min_RGB_Tile, Median_Tile);


    // Subtract Median_Tile from image_tile to get the recovered radiance (capping values to [0, 255])
    // Store recovered radiance in image_tile and return
    // Tuning parameter = kh (between 0 to 100)
    // Higher kh == more of the "haze" subtracted

    INT_TYPE kh = 96;

    for (int i = 0; i < TILE_HEIGHT; i++)
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            // if (i == DEBUG_ROW && j == DEBUG_COL)
            // {
            //     std::cout << "tile BGR = " << image_tile[i][j][0] << "," << image_tile[i][j][1] << "," << image_tile[i][j][2] << std::endl;
            //     std::cout << "median tile = " << Median_Tile[i][j] << std::endl;
            // }

            LONG_SINT_TYPE tmp0 = (100 * image_tile[i][j][0] - (kh * Median_Tile[i][j])) * 255 / (100 * (256 - Median_Tile[i][j]));
            if (tmp0 < 0)
                image_tile[i][j][0] = 0;
            else if (tmp0 > 255)
                image_tile[i][j][0] = 255;
            else
                image_tile[i][j][0] = tmp0;

            LONG_SINT_TYPE tmp1 = (100 * image_tile[i][j][1] - (kh * Median_Tile[i][j])) * 255 / (100 * (256 - Median_Tile[i][j]));
            if (tmp1 < 0)
                image_tile[i][j][1] = 0;
            else if (tmp1 > 255)
                image_tile[i][j][1] = 255;
            else
                image_tile[i][j][1] = tmp1;

            LONG_SINT_TYPE tmp2 = (100 * image_tile[i][j][2] - (kh * Median_Tile[i][j])) * 255 / (100 * (256 - Median_Tile[i][j]));
            if (tmp2 < 0)
                image_tile[i][j][2] = 0;
            else if (tmp2 > 255)
                image_tile[i][j][2] = 255;
            else
                image_tile[i][j][2] = tmp2;

            // if (i == DEBUG_ROW && j == DEBUG_COL)
            // {
            //     std::cout << "AFTER: " << std::endl;
            //     std::cout << "tile BGR = " << image_tile[i][j][0] << "," << image_tile[i][j][1] << "," << image_tile[i][j][2] << std::endl;
            // }
        }
}