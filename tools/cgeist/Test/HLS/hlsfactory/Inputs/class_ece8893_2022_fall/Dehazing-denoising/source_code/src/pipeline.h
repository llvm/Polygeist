#ifndef __DCL_H__
#define __DCL_H__

#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>
#include <ap_fixed.h>

typedef ap_uint<8> INT_TYPE;
typedef ap_uint<16> INT16_TYPE;
typedef ap_uint<20> LONG_INT_TYPE;
typedef ap_int<20> LONG_SINT_TYPE;

#define IMG_HEIGHT 384 // height
#define IMG_WIDTH 512  // width
// #define IMG_HEIGHT 512 // height
// #define IMG_WIDTH 384  // width

#define IN_TILE_HEIGHT 72 // height   64 + 1*2 + 3*2
#define IN_TILE_WIDTH 72  // width   64 + 1*2 + 3*2
// #define IN_TILE_HEIGHT 392                // height   16 + 1*2 + 3*2
// #define IN_TILE_WIDTH 72                 // width   16 + 1*2 + 3*2

#define TILE_HEIGHT (IN_TILE_HEIGHT - 2)
#define TILE_WIDTH (IN_TILE_WIDTH - 2)

#define N_TILES_X IMG_WIDTH / (IN_TILE_WIDTH - 8)
#define N_TILES_Y IMG_HEIGHT / (IN_TILE_HEIGHT - 8)

#define COLOR_INT 256

// #define DEBUG_ROW 155
// #define DEBUG_COL 299

/* IP Blocks */
void img_pipeline(INT_TYPE in[IMG_HEIGHT][IMG_WIDTH],
                  INT_TYPE out[IMG_HEIGHT][IMG_WIDTH][3]);

void debayer(INT_TYPE in[IN_TILE_HEIGHT][IN_TILE_WIDTH],
             INT_TYPE out[TILE_HEIGHT][TILE_WIDTH][3],
             int ti,
             int tj,
             LONG_INT_TYPE blue_hist[COLOR_INT],
             LONG_INT_TYPE green_hist[COLOR_INT],
             LONG_INT_TYPE red_hist[COLOR_INT]);

void tile_dehaze(INT_TYPE image_tile[TILE_HEIGHT][TILE_WIDTH][3]);

void update_vmin_vmax(
    LONG_INT_TYPE blue_hist[COLOR_INT],
    LONG_INT_TYPE green_hist[COLOR_INT],
    LONG_INT_TYPE red_hist[COLOR_INT],
    INT_TYPE vmin[3],
    INT_TYPE vmax[3]);

void wb_apply_transform_tile(
    INT_TYPE image_tile[TILE_HEIGHT][TILE_WIDTH][3],
    INT_TYPE vmin[3],
    INT_TYPE vmax[3]);

/* Util Functions */
void load_input_tile_from_DRAM(
    INT_TYPE in_tile[IN_TILE_HEIGHT][IN_TILE_WIDTH],
    INT_TYPE in_img[IMG_HEIGHT][IMG_WIDTH],
    int ti,
    int tj);
void store_output_tile_to_DRAM(
    INT_TYPE out_fm_buf[TILE_HEIGHT][TILE_WIDTH][3],
    INT_TYPE out_fm[IMG_HEIGHT][IMG_WIDTH][3],
    int ti,
    int tj);

#endif
