#pragma once

#include "../util.h"

namespace conv1
{
    void tiled_conv(
        fm_t output_feature_map[64][112][112],
        const fm_t input_feature_map[3][224][224],
        const wt_t layer_weights[64][3][7][7],
        const wt_t layer_bias[64]
    );
}
