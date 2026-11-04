
// Command line : python pseudo.py 64 2**255-19

#include <stdint.h>
#include <stdio.h>

#define sspint int64_t
#define spint uint64_t
#define udpint __uint128_t
#define dpint __uint128_t

#define Wordlength 64
#define Nlimbs 5
#define Radix 51
#define Nbits 255
#define Nbytes 32

#define MERSENNE
#define MULBYINT


static void inline modmul(const spint *a, const spint *b, spint *c) {
  dpint t = 0;
  spint ma1 = a[1] * (spint)0x13;
  spint ma2 = a[2] * (spint)0x13;
  spint ma3 = a[3] * (spint)0x13;
  spint ma4 = a[4] * (spint)0x13;
  spint carry;
  spint s;
  spint mask = ((spint)1 << 51u) - (spint)1;
  t += (dpint)ma1 * (dpint)b[4];
  t += (dpint)ma2 * (dpint)b[3];
  t += (dpint)ma3 * (dpint)b[2];
  t += (dpint)ma4 * (dpint)b[1];
  t += (dpint)a[0] * (dpint)b[0];
  spint v0 = (spint)t & mask;
  t = t >> 51u;
  t += (dpint)ma2 * (dpint)b[4];
  t += (dpint)ma3 * (dpint)b[3];
  t += (dpint)ma4 * (dpint)b[2];
  t += (dpint)a[0] * (dpint)b[1];
  t += (dpint)a[1] * (dpint)b[0];
  spint v1 = (spint)t & mask;
  t = t >> 51u;
  t += (dpint)ma3 * (dpint)b[4];
  t += (dpint)ma4 * (dpint)b[3];
  t += (dpint)a[0] * (dpint)b[2];
  t += (dpint)a[1] * (dpint)b[1];
  t += (dpint)a[2] * (dpint)b[0];
  spint v2 = (spint)t & mask;
  t = t >> 51u;
  t += (dpint)ma4 * (dpint)b[4];
  t += (dpint)a[0] * (dpint)b[3];
  t += (dpint)a[1] * (dpint)b[2];
  t += (dpint)a[2] * (dpint)b[1];
  t += (dpint)a[3] * (dpint)b[0];
  spint v3 = (spint)t & mask;
  t = t >> 51u;
  t += (dpint)a[0] * (dpint)b[4];
  t += (dpint)a[1] * (dpint)b[3];
  t += (dpint)a[2] * (dpint)b[2];
  t += (dpint)a[3] * (dpint)b[1];
  t += (dpint)a[4] * (dpint)b[0];
  spint v4 = (spint)t & mask;
  t = t >> 51u;
  // second reduction pass

  spint ut = (spint)t;
  ut *= 0x13;
  s = v0 + ((spint)ut & mask);
  c[0] = (spint)(s & mask);
  carry = (s >> 51) + (spint)(ut >> 51);
  c[1] = v1 + carry;
  c[2] = v2;
  c[3] = v3;
  c[4] = v4;
}

