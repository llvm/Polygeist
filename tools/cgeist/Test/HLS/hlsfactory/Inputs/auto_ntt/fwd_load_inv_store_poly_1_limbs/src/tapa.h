#ifndef TAPA_TAPA_H_
#define TAPA_TAPA_H_

// added a hard definition because we never
// perform C simulation
#define TAPA_TARGET_ (XILINX_HLS)

#if !defined(TAPA_TARGET_)

#include "tapa/host/tapa.h"

#elif TAPA_TARGET_ == XILINX_HLS

#include "tapa/xilinx/hls/tapa.h"

#endif

#include "tapa/traits.h"

#endif  // TAPA_TAPA_H_
