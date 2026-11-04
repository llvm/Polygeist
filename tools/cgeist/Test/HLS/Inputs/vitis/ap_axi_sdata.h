// Minimal stand-in for the Vitis HLS ap_axi_sdata.h, for cgeist tests only.
#ifndef CGEIST_TEST_AP_AXI_SDATA_H
#define CGEIST_TEST_AP_AXI_SDATA_H

#ifdef __cplusplus
#include "ap_int.h"

template <int D, int U, int TI, int TD> struct ap_axis {
  ap_int<D> data;
  ap_uint<(D + 7) / 8> keep;
  ap_uint<(D + 7) / 8> strb;
  ap_uint<U> user;
  ap_uint<1> last;
  ap_uint<TI> id;
  ap_uint<TD> dest;
};
template <int D, int U, int TI, int TD> struct ap_axiu {
  ap_uint<D> data;
  ap_uint<(D + 7) / 8> keep;
  ap_uint<(D + 7) / 8> strb;
  ap_uint<U> user;
  ap_uint<1> last;
  ap_uint<TI> id;
  ap_uint<TD> dest;
};
namespace hls {
template <typename T, int U = 0, int TI = 0, int TD = 0> struct axis {
  T data;
  ap_uint<(sizeof(T) + 0)> keep;
  ap_uint<(sizeof(T) + 0)> strb;
  ap_uint<U + 1> user;
  ap_uint<1> last;
  ap_uint<TI + 1> id;
  ap_uint<TD + 1> dest;
};
} // namespace hls
#endif // __cplusplus
#endif
