#ifndef TAPA_XILINX_HLS_TAPA_H_
#define TAPA_XILINX_HLS_TAPA_H_

#include "tapa/base/tapa.h"

#ifdef __SYNTHESIS__

#include "tapa/xilinx/hls/mmap.h"
#include "tapa/xilinx/hls/stream.h"
#include "tapa/xilinx/hls/task.h"
#include "tapa/xilinx/hls/util.h"
#include "tapa/xilinx/hls/vec.h"
#ifdef TAPA_BUFFER_SUPPORT
#include "tapa/xilinx/hls/buffer.h"
#endif

#else  // __SYNTHESIS__

#include "tapa/host/tapa.h"

#endif

#endif  // TAPA_XILINX_HLS_TAPA_H_
