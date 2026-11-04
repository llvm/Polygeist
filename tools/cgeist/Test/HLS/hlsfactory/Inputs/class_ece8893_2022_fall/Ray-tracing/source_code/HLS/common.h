#ifndef _COMMON_H_

#define _COMMON_H_

#include <limits>
#include <cmath>
#include "hls_math.h"
#include "ap_fixed.h"
#include "geometry.h"

static const fixed_t kInfinity = (fixed_t)pow(2,14);//std::numeric_limits<fixed_t>::max();
// NOTE: This decides the accuracy/precision of intersection checks
static const fixed_t kEpsilon = (fixed_t)pow(2,-15);//(fixed_t)1e-5;

fixed_t customNorm3(fixed_t x[3]);

void customNormalize3(fixed_t x[3]);

void customMultVecMatrix(fixed_t src[3], fixed_t dst[3], fixed_t x[4][4]);

void customMultDirMatrix(fixed_t src[3], fixed_t dst[3], fixed_t x[4][4]);

/*
* Function to implement cross product
* result = in1 x in2
*/
void customCrossProduct(fixed_t in1[3], fixed_t in2[3], fixed_t result[3]);

/*
* Function to implement dot product
* result = in1 . in2
*/
void customDotProduct(fixed_t in1[3], fixed_t in2[3], fixed_t &result);

/*
* Function to implement subration
* result = in1 - in2
*/
void customSubtract(fixed_t in1[3], fixed_t in2[3], fixed_t result[3]);


void copy3(fixed_t in[3], fixed_t out[3]);


void copy2(fixed_t in[2], fixed_t out[2]);

fixed_t customFmod(fixed_t x);

void customDivide(fixed_t &in1, fixed_t &in2, fixed_t &result);

// COPY FUNCTIONS:

void copyP(
    fixed_t P1_DRAM[NUM_TRIS][3],
    fixed_t P2_DRAM[NUM_TRIS][3],
    fixed_t P3_DRAM[NUM_TRIS][3],
    fixed_t P1[NUM_TRIS][3],
    fixed_t P2[NUM_TRIS][3],
    fixed_t P3[NUM_TRIS][3]);

void copyCTW(
    fixed_t cameraToWorld_DRAM[4][4],
    fixed_t cameraToWorld[4][4]);

void copyTex(
    fixed_t texCoordinates_DRAM[NUM_TRIS * 3][2],
    fixed_t texCoordinates[NUM_TRIS * 3][2]);

#endif
