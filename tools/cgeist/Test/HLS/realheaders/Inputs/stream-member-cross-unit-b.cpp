// Second unit of ../stream-member-cross-unit.cpp: uses operator>> and calls prod.
#include <hls_stream.h>
#include <ap_fixed.h>
#include <array>
typedef std::array<ap_fixed<16, 3>, 2> tok_t;
void prod(tok_t *in, hls::stream<tok_t> &s);
void cons(hls::stream<tok_t> &s, tok_t *out) {
  for (int i = 0; i < 4; i++)
    s >> out[i];
}
void top(tok_t *in, tok_t *out) {
  hls::stream<tok_t> s;
  prod(in, s);
  cons(s, out);
}
