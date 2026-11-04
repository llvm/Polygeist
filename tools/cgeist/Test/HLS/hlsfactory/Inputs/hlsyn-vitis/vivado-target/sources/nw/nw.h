#ifndef TOP_H
#define TOP_H


#include <string.h>
#include "ap_fixed.h"
#include <complex>
#include <hls_stream.h>
#include <iomanip>
#include <iostream>
#include <cmath>
#include "hls_vector.h"

void needwun(char SEQA[128],char SEQB[128],char alignedA[256],char alignedB[256],int M[16641],char ptr[16641]);

#endif
