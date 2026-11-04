/*===============================================================*/
/*                                                               */
/*                       optical_flow.h                          */
/*                                                               */
/*             Hardware function for optical flow                */
/*                                                               */
/*===============================================================*/

#ifndef __OPTICAL_FLOW_H__
#define __OPTICAL_FLOW_H__

#include "../host/typedefs.h"

// convolution filters
const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};

// top-level function 
#pragma SDS data access_pattern(frames:SEQUENTIAL, outputs:SEQUENTIAL)
void optical_flow(frames_t   frames[MAX_HEIGHT][MAX_WIDTH],
                  velocity_t outputs[MAX_HEIGHT][MAX_WIDTH]);

#endif
