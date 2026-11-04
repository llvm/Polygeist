#include "dcl.h"

void get_reduced_terms(spint_vec& src, spint_red_vec &dst)
{
  for (int i = 1; i < Nlimbs; i++) {
#pragma HLS unroll
    dst[i] = src[i] * ReductionFactor;
  }
}

void mod_mul(spint_vec &x, spint_vec &y, spint_vec &z) {
#pragma HLS interface mode = axis port = x // offset = slave bundle = mem1
#pragma HLS interface mode = axis port = y // offset = slave bundle = mem2
#pragma HLS interface mode = axis port = z // offset = slave bundle = mem3
#pragma HLS interface s_axilite port = return

  dpint t = 0;
  spint mask = ((spint)1 << Radix) - (spint)1;

  spint_vec x_buf = x;
  spint_vec y_buf = y;
  spint_vec z_buf;
  
  spint_vec v;
  spint_red_vec x_buf_reduced;

  get_reduced_terms(x_buf, x_buf_reduced);

REDUCTION:
  for (int i = 0; i < Nlimbs; i++) {
#pragma HLS unroll
    dpint tt = 0;

    // Calculate high terms (terms that are reduced modulo 2^255 - 19)
  HIGH_TERMS:
    for (int j = 1 + i; j < Nlimbs; j++) {
#pragma HLS unroll
      tt += x_buf_reduced[j] * y_buf[Nlimbs - j + i];
    }

  // Calculate direct terms (terms that aren't reduced)
  DIRECT_TERMS:
    for (int j = 0; j < i + 1; j++) {
#pragma HLS unroll
      t += x_buf[j] * y_buf[i - j];
    }

    // Add the high terms to the direct terms
    t += tt;

    v[i] = t & mask;
    t = t >> Radix;
  }

  // Second reduction pass
  udpint ut = t;

  // Reduce the overflow bits of the last term
  if (LastLimbOverflows) {
    ut = (dpint)(ut << OverflowBits) +
         (spint)(v[Nlimbs - 1] >> (FinalLimbRadix));

    // Get the non-overflow bits
    v[Nlimbs - 1] &= (spint(1) << FinalLimbRadix) - 1;
  }
  // Reduce the overflow bits of the last term
  ut *= CongruentFactor;

  // Add the reduced bits to the first term
  dpint sum = v[0] + (ut & mask);

  z_buf[0] = sum & mask;

  // Get the carry bits from the sum and the overflow bits
  spint carry = (sum >> Radix) + (spint)(ut >> Radix);

  z_buf[CarryIndex] = v[CarryIndex] + carry;

// Set the rest of the limbs
SET_LIMBS:
  for (int i = 2; i < Nlimbs; i++) {
#pragma HLS unroll
    z_buf[i] = v[i];
  }

  // Send the result to the output
  z = z_buf;
}
