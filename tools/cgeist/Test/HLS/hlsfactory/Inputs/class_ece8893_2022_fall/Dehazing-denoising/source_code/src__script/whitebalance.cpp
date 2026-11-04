#include <iostream>
#include "pipeline.h"


void update_vmin_vmax(
    LONG_INT_TYPE blue_hist[COLOR_INT],
    LONG_INT_TYPE green_hist[COLOR_INT],
    LONG_INT_TYPE red_hist[COLOR_INT],
    INT_TYPE vmin[3],
    INT_TYPE vmax[3]
)
{

    LONG_INT_TYPE pixel_cnt[3] = {0,0,0};
    #pragma HLS ARRAY_PARTITION variable = pixel_cnt complete dim = 1
    LONG_INT_TYPE pixel_cnt2[3] = {0,0,0};
    #pragma HLS ARRAY_PARTITION variable = pixel_cnt2 complete dim = 1
    
    LONG_INT_TYPE N = IMG_HEIGHT * IMG_WIDTH;
    INT_TYPE factor = 20;

    for(int i=0; i < COLOR_INT; i++)
    {
        if(pixel_cnt[0] <= N/factor){
            pixel_cnt[0] += blue_hist[i];
            vmin[0] = i;
        }
        if(pixel_cnt[1] <= N/factor){
            pixel_cnt[1] += green_hist[i];
            vmin[1] = i;
        }
        if(pixel_cnt[2] <= N/factor){
            pixel_cnt[2] += red_hist[i];
            vmin[2] = i;
        }

        if(pixel_cnt2[0] <= (N*factor-N)/factor){
            pixel_cnt2[0] += blue_hist[i];
            vmax[0] = i;
        }
        if(pixel_cnt2[1] <= (N*factor-N)/factor){
            pixel_cnt2[1] += green_hist[i];
            vmax[1] = i;
        }
        if(pixel_cnt2[2] <= (N*factor-N)/factor){
            pixel_cnt2[2] += red_hist[i];
            vmax[2] = i;
        }
    }
}

void wb_apply_transform_tile(INT_TYPE image_tile[TILE_HEIGHT][TILE_WIDTH][3], INT_TYPE vmin[3], INT_TYPE vmax[3])
{
    //white balance the image
    for (int i = 0; i < TILE_HEIGHT; i++)
    {
        for (int j = 0; j < TILE_WIDTH; j++)
        {

            if(image_tile[i][j][0] > vmax[0])
                image_tile[i][j][0] = vmax[0];

            if(image_tile[i][j][0] < vmin[0])
                image_tile[i][j][0] = vmin[0];

            if(image_tile[i][j][1] > vmax[1])
                image_tile[i][j][1] = vmax[1];

            if(image_tile[i][j][1] < vmin[1])
                image_tile[i][j][1] = vmin[1];

            if(image_tile[i][j][2] > vmax[2])
                image_tile[i][j][2] = vmax[2];

            if(image_tile[i][j][2] < vmin[2])
                image_tile[i][j][2] = vmin[2];

            //APPLY AFFINE TRANSFORMATION

            //image_tile[i][j][0] = ((image_tile[i][j][0] - vmin[0]) * COLOR_INT) / (vmax[0] - vmin[0]);
            //image_tile[i][j][1] = ((image_tile[i][j][1] - vmin[1]) * COLOR_INT) / (vmax[1] - vmin[1]);
            //image_tile[i][j][2] = ((image_tile[i][j][2] - vmin[2]) * COLOR_INT) / (vmax[2] - vmin[2]);

            LONG_SINT_TYPE image_tile_0 = ((image_tile[i][j][0] - vmin[0]) * COLOR_INT) / (vmax[0] - vmin[0]);
            LONG_SINT_TYPE image_tile_1 = ((image_tile[i][j][1] - vmin[1]) * COLOR_INT) / (vmax[1] - vmin[1]);
            LONG_SINT_TYPE image_tile_2 = ((image_tile[i][j][2] - vmin[2]) * COLOR_INT) / (vmax[2] - vmin[2]);

            //APPLY FLOOR AND CEIL

            if(image_tile_0 >= COLOR_INT)
                image_tile[i][j][0] = COLOR_INT - 1;
            else if(image_tile_0 < 0)
                image_tile[i][j][0] = 0;
            else
                image_tile[i][j][0] = image_tile_0;

            if(image_tile_1 >= COLOR_INT)
                image_tile[i][j][1] = COLOR_INT - 1;
            else if(image_tile_1 < 0)
                image_tile[i][j][1] = 0;
            else
                image_tile[i][j][1] = image_tile_1;

            if(image_tile_2 >= COLOR_INT)
                image_tile[i][j][2] = COLOR_INT - 1;
            else if(image_tile_2 < 0)
                image_tile[i][j][2] = 0;
            else
                image_tile[i][j][2] = image_tile_2;
        }
    }

}