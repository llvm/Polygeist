#pragma once

#include "../util.h"

/* Max kernel template
 *
 * Template parameters:
 *  D: Dimension of the array
 *
 * @param: out: Index of the maximum value
 * @param: x: Array of values
 */
template <int D, typename fixp_t >
int max_idx(fixp_t x[D])
{
    int ans = 0;
    for (int i = 0; i < D; i++)
        if (x[i] > x[ans])
            ans = i;

    return ans;
}
