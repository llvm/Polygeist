#pragma once

const int OUT_BUF_DEPTH = 16;
const int IN_BUF_DEPTH = 64;
const int KERNEL_HEIGHT = 3;
const int KERNEL_WIDTH = 3;
const int STRIDE = 1;
const int PADDING = 1;
const int TILE_HEIGHT = 7;
const int TILE_WIDTH = 7;

const int IN_BUF_HEIGHT = 2*TILE_HEIGHT + 2 * PADDING;
const int IN_BUF_WIDTH = 2*TILE_WIDTH + 2 * PADDING;
const int OUT_BUF_HEIGHT = STRIDE == 1 ? TILE_HEIGHT : TILE_HEIGHT >> 1;
const int OUT_BUF_WIDTH = STRIDE == 1 ? TILE_WIDTH : TILE_WIDTH >> 1;
