// Third unit of ../stream-member-cross-unit.cpp: instantiates operator<< and write.
#include <hls_stream.h>
#include <ap_fixed.h>
#include <array>
typedef std::array<ap_fixed<16, 3>, 2> tok_t;
void again(tok_t v, hls::stream<tok_t> &s) { s.write(v); s << v; }
