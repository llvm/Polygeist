#ifndef TAPA_STREAM_H_
#define TAPA_STREAM_H_

#if !defined(TAPA_TARGET_)

#include "tapa/host/stream.h"

#elif TAPA_TARGET_ == XILINX_HLS

#include "tapa/xilinx/hls/stream.h"

#endif

#endif  // TAPA_STREAM_H_
