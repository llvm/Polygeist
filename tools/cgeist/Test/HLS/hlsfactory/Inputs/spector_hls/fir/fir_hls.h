#include "params.h"
#include <iostream> // for debugging

#define PRAGMA_SUB(x) _Pragma(#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

typedef unsigned char uchar;
typedef unsigned short ushort;

using namespace std;

void fir_hls(
    float dataPtr[INPUT_SIZE],
    float filterPtr[FILTER_SIZE],
    float resultPtr[RESULT_SIZE],
    unsigned int totalInputLength,
    const int paddedSingleInputLength,
    const int totalFiltersLength);

void fir_fpga(
    float dataPtr[INPUT_SIZE],
    float filterPtr[FILTER_SIZE],
    float resultPtr[RESULT_SIZE],
    unsigned int totalInputLength,
    const int paddedSingleInputLength,
    const int totalFiltersLength,
    unsigned int tid);
