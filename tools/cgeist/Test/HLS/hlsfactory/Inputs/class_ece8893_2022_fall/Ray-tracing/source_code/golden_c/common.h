#ifndef _COMMON_H_

#define _COMMON_H_

#include <limits>
#include <math.h>
#include "geometry.h"

static const float kInfinity = std::numeric_limits<float>::max();
static const float kEpsilon = 1e-8;

float customNorm3(float x[3]);

void customNormalize3(float x[3]);

inline float customDeg2Rad(const float deg)
{
    return deg * M_PI / 180;
}

void customMultVecMatrix(float src[3], float dst[3], float x[4][4]);

void customMultDirMatrix(float src[3], float dst[3], float x[4][4]);

/*
* Function to implement cross product
* result = in1 x in2
*/
void customCrossProduct(float in1[3], float in2[3], float result[3]);

/*
* Function to implement dot product
* result = in1 . in2
*/
void customDotProduct(float in1[3], float in2[3], float &result);

/*
* Function to implement subration
* result = in1 - in2
*/
void customSubtract(float in1[3], float in2[3], float result[3]);


void copy3(float in[3], float out[3]);


void copy2(float in[2], float out[2]);

#endif
