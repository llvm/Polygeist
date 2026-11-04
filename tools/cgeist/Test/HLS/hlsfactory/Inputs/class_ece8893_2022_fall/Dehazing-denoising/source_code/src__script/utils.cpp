#include "pipeline.h"

void load_input_tile_from_DRAM(
    INT_TYPE in_tile[IN_TILE_HEIGHT][IN_TILE_WIDTH],
    INT_TYPE in_img[IMG_HEIGHT][IMG_WIDTH],
    int ti,
    int tj)
{
    const int height_offset = (ti * (IN_TILE_HEIGHT - 8)); // OUT_BUF is intended, not a typo.
    const int width_offset = (tj * (IN_TILE_WIDTH - 8));

INPUT_BUFFER_HEIGHT:
    for (int i = 0; i < IN_TILE_HEIGHT; i++)
    {
    INPUT_BUFFER_WIDTH:
        for (int j = 0; j < IN_TILE_WIDTH; j++)
        {
            if (!((height_offset + i - 4 < 0) || (width_offset + j - 4 < 0) || (height_offset + i - 4 > IMG_HEIGHT) || (width_offset + j - 4 > IMG_WIDTH)))
            {
                in_tile[i][j] = in_img[height_offset + i - 4][width_offset + j - 4];
            }
        }
    }
}

void store_output_tile_to_DRAM(
    INT_TYPE out_fm_buf[TILE_HEIGHT][TILE_WIDTH][3],
    INT_TYPE out_fm[IMG_HEIGHT][IMG_WIDTH][3],
    int ti,
    int tj)
{
    const int height_offset = (ti * (TILE_HEIGHT - 6));
    const int width_offset = (tj * (TILE_WIDTH - 6));

OUTPUT_BUFFER_HEIGHT:
    for (int i = 3; i < TILE_HEIGHT - 3; i++)
    {
    OUTPUT_BUFFER_WIDTH:
        for (int j = 3; j < TILE_WIDTH - 3; j++)
        {
            out_fm[height_offset + i - 3][width_offset + j - 3][0] = out_fm_buf[i][j][0];
        }
    }

    for (int i = 3; i < TILE_HEIGHT - 3; i++)
    {
        for (int j = 3; j < TILE_WIDTH - 3; j++)
        {
            out_fm[height_offset + i - 3][width_offset + j - 3][1] = out_fm_buf[i][j][1];
        }
    }

    for (int i = 3; i < TILE_HEIGHT - 3; i++)
    {
        for (int j = 3; j < TILE_WIDTH - 3; j++)
        {
            out_fm[height_offset + i - 3][width_offset + j - 3][2] = out_fm_buf[i][j][2];
        }
    }
}