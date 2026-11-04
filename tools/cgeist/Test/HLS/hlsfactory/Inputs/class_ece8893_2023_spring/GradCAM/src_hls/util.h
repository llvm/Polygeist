#pragma once

#include <iostream>

#ifdef CSIM_DEBUG
    typedef float fm_t;
    typedef float wt_t;
#else
#include <ap_fixed.h>
    typedef ap_fixed<16,3> fm_t;
    typedef ap_fixed<16,3> wt_t;
#endif

typedef int dim_t;

typedef struct {
    dim_t depth;
    dim_t height;
    dim_t width;
} fm_dims_s;


#define VAR_NAME(x) #x
