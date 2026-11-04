#pragma once

const int OUT_BUF_DEPTH = 8;
const int IN_BUF_DEPTH = 3;
const int KERNEL_HEIGHT = 7;
const int KERNEL_WIDTH = 7;
const int STRIDE = 2;
const int PADDING = 3;
const int TILE_HEIGHT = 32;
const int TILE_WIDTH = 32;
const int IN_FM_DEPTH = 3;
const int IN_FM_HEIGHT = 224;
const int IN_FM_WIDTH = 224;
const int OUT_FM_HEIGHT = 112;
const int OUT_FM_WIDTH = 112;
const int OUT_FM_DEPTH = 64;
const int KERNEL_GRPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;
const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;
const int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;
const bool inplace_residual = false;

const int MARGIN = 2 * PADDING;
const int IN_BUF_HEIGHT = TILE_HEIGHT + MARGIN;
const int IN_BUF_WIDTH = TILE_WIDTH + MARGIN;
const int OUT_BUF_HEIGHT = STRIDE == 1 ? TILE_HEIGHT : TILE_HEIGHT >> 1;
const int OUT_BUF_WIDTH = STRIDE == 1 ? TILE_WIDTH : TILE_WIDTH >> 1;
const bool relu = true;
