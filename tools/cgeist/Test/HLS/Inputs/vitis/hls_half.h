// hls-est "exactclass" stand-in for the Vitis HLS hls_half.h: half is the
// 16-bit IEEE binary16 type (cgeist lowers _Float16 to f16).
#ifndef HLSEST_EXACTCLASS_HLS_HALF_H
#define HLSEST_EXACTCLASS_HLS_HALF_H
typedef _Float16 half;
#endif
