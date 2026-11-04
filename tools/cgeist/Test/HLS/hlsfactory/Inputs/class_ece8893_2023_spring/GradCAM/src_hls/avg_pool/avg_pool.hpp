#pragma once

#include "../util.h"

namespace avg_pool {


const int IN_FM_DEPTH = 512;
const int IN_FM_HEIGHT = 7;
const int IN_FM_WIDTH = 7;

void avg_pool(
    fm_t in[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    fm_t out[IN_FM_DEPTH]
);

}
