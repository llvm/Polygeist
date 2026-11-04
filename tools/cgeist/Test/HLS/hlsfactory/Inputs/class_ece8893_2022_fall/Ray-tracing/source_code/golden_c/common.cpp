#include "common.h"

void customMultVecMatrix(float src[3], float dst[3], float x[4][4])
{
     float val[4];

    for (int i = 0; i < 4; ++i)
    {
        float temp_val = 0;
        for (int j = 0; j < 3; ++j)
        {
            temp_val += src[j] * x[j][i];
        }
        val[i] = temp_val + x[3][i];
    }

    for (int i = 0; i < 3; ++i)
    {
        dst[i] = val[i] / val[3];
    }
}

void customMultDirMatrix(float src[3], float dst[3], float x[4][4])
{
    float val[3];

    for (int i = 0; i < 3; ++i)
    {
        float temp_val = 0;
        for (int j = 0; j < 3; ++j)
        {
            temp_val += src[j] * x[j][i];
        }
        val[i] = temp_val;
    }

    for (int i = 0; i < 3; ++i)
    {
        dst[i] = val[i];
    }
}

/*
* Function to implement cross product
* result = in1 x in2
*/
void customCrossProduct(float in1[3], float in2[3], float result[3])
{
    for (int i = 0; i < 3; ++i)
    {
        result[i] = in1[(i+1)%3] * in2[(i+2)%3] - in1[(i+2)%3] * in2[(i+1)%3];
    }
}

/*
* Function to implement dot product
* result = in1 . in2
*/
void customDotProduct(float in1[3], float in2[3], float &result)
{
    float temp_val = 0;
    for (int i = 0; i < 3; ++i)
    {
        temp_val += in1[i] * in2[i];
    }
    result = temp_val;
}

/*
* Function to implement subration
* result = in1 - in2
*/
void customSubtract(float in1[3], float in2[3], float result[3])
{
    for (int i = 0; i < 3; ++i)
    {
        result[i] = in1[i] - in2[i];
    }
}

void copy3(float in[3], float out[3])
{
    for (int i = 0; i < 3; ++i)
    {
        out[i] = in[i];
    }
}

void copy2(float in[2], float out[2])
{
    for (int i = 0; i < 2; ++i)
    {
        out[i] = in[i];
    }
}

float customNorm3(float x[3])
{
    float temp_val;
    customDotProduct(x, x, temp_val);
    return temp_val;
}

void customNormalize3(float x[3])
{
    float n = customNorm3(x);
    if (n > 0.0)
    {
        float factor = 1 / sqrt(n);
        for (int i = 0; i < 3; ++i)
        {
            x[i] *= factor;
        }
    }
}

