#ifndef _TB_COMMON_H_
#define _TB_COMMON_H_

#include <algorithm>

inline float customClamp(const float &lo, const float &hi, const float &v)
{
    return std::max(lo, std::min(hi, v));
}

void customInverse(float mat[4][4], float inverseMat[4][4]);

void customCopy44(float in[4][4], float out[4][4]);

#endif