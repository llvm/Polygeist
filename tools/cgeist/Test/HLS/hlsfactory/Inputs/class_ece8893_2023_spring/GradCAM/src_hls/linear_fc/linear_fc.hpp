#pragma once

#include "../util.h"

namespace linear_fc {

void linear_fc(
    const fm_t in[512],
    fm_t out[1000],
    const wt_t weights[1000][512],
    const wt_t biases[1000]
);

}
