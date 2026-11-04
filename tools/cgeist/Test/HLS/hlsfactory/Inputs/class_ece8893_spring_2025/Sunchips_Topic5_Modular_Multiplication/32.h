
// Command line : python pseudo.py 32 2**255-19

#include <stdint.h>
#include <stdio.h>

#define sspint int32_t
#define spint uint32_t
#define udpint uint64_t
#define dpint uint64_t

#define Wordlength 32
#define Nlimbs 9
#define Radix 29
#define Nbits 255
#define Nbytes 32

#define MERSENNE
#define MULBYINT


static void modmul(const spint *a, const spint *b, spint *c) {
  dpint t = 0;
  dpint tt;
  spint lo;
  spint hi;
  spint carry;
  spint s;
  spint mask = ((spint)1 << 29u) - (spint)1;
  tt = (dpint)a[1] * (dpint)b[8];
  tt += (dpint)a[2] * (dpint)b[7];
  tt += (dpint)a[3] * (dpint)b[6];
  tt += (dpint)a[4] * (dpint)b[5];
  tt += (dpint)a[5] * (dpint)b[4];
  tt += (dpint)a[6] * (dpint)b[3];
  tt += (dpint)a[7] * (dpint)b[2];
  tt += (dpint)a[8] * (dpint)b[1];
  lo = (spint)tt & mask;
  t += (dpint)lo * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[0];
  spint v0 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[2] * (dpint)b[8];
  tt += (dpint)a[3] * (dpint)b[7];
  tt += (dpint)a[4] * (dpint)b[6];
  tt += (dpint)a[5] * (dpint)b[5];
  tt += (dpint)a[6] * (dpint)b[4];
  tt += (dpint)a[7] * (dpint)b[3];
  tt += (dpint)a[8] * (dpint)b[2];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[1];
  t += (dpint)a[1] * (dpint)b[0];
  spint v1 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[3] * (dpint)b[8];
  tt += (dpint)a[4] * (dpint)b[7];
  tt += (dpint)a[5] * (dpint)b[6];
  tt += (dpint)a[6] * (dpint)b[5];
  tt += (dpint)a[7] * (dpint)b[4];
  tt += (dpint)a[8] * (dpint)b[3];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[2];
  t += (dpint)a[1] * (dpint)b[1];
  t += (dpint)a[2] * (dpint)b[0];
  spint v2 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[4] * (dpint)b[8];
  tt += (dpint)a[5] * (dpint)b[7];
  tt += (dpint)a[6] * (dpint)b[6];
  tt += (dpint)a[7] * (dpint)b[5];
  tt += (dpint)a[8] * (dpint)b[4];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[3];
  t += (dpint)a[1] * (dpint)b[2];
  t += (dpint)a[2] * (dpint)b[1];
  t += (dpint)a[3] * (dpint)b[0];
  spint v3 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[5] * (dpint)b[8];
  tt += (dpint)a[6] * (dpint)b[7];
  tt += (dpint)a[7] * (dpint)b[6];
  tt += (dpint)a[8] * (dpint)b[5];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[4];
  t += (dpint)a[1] * (dpint)b[3];
  t += (dpint)a[2] * (dpint)b[2];
  t += (dpint)a[3] * (dpint)b[1];
  t += (dpint)a[4] * (dpint)b[0];
  spint v4 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[6] * (dpint)b[8];
  tt += (dpint)a[7] * (dpint)b[7];
  tt += (dpint)a[8] * (dpint)b[6];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[5];
  t += (dpint)a[1] * (dpint)b[4];
  t += (dpint)a[2] * (dpint)b[3];
  t += (dpint)a[3] * (dpint)b[2];
  t += (dpint)a[4] * (dpint)b[1];
  t += (dpint)a[5] * (dpint)b[0];
  spint v5 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[7] * (dpint)b[8];
  tt += (dpint)a[8] * (dpint)b[7];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[6];
  t += (dpint)a[1] * (dpint)b[5];
  t += (dpint)a[2] * (dpint)b[4];
  t += (dpint)a[3] * (dpint)b[3];
  t += (dpint)a[4] * (dpint)b[2];
  t += (dpint)a[5] * (dpint)b[1];
  t += (dpint)a[6] * (dpint)b[0];
  spint v6 = (spint)t & mask;
  t = t >> 29u;
  tt = (dpint)a[8] * (dpint)b[8];
  lo = (spint)tt & mask;
  t += (dpint)(spint)(lo + hi) * (dpint)0x4c0;
  hi = (spint)(tt >> 29u);
  t += (dpint)a[0] * (dpint)b[7];
  t += (dpint)a[1] * (dpint)b[6];
  t += (dpint)a[2] * (dpint)b[5];
  t += (dpint)a[3] * (dpint)b[4];
  t += (dpint)a[4] * (dpint)b[3];
  t += (dpint)a[5] * (dpint)b[2];
  t += (dpint)a[6] * (dpint)b[1];
  t += (dpint)a[7] * (dpint)b[0];
  spint v7 = (spint)t & mask;
  t = t >> 29u;
  t += (dpint)a[0] * (dpint)b[8];
  t += (dpint)a[1] * (dpint)b[7];
  t += (dpint)a[2] * (dpint)b[6];
  t += (dpint)a[3] * (dpint)b[5];
  t += (dpint)a[4] * (dpint)b[4];
  t += (dpint)a[5] * (dpint)b[3];
  t += (dpint)a[6] * (dpint)b[2];
  t += (dpint)a[7] * (dpint)b[1];
  t += (dpint)a[8] * (dpint)b[0];
  t += (dpint)hi * (dpint)0x4c0;
  spint v8 = (spint)t & mask;
  t = t >> 29u;
  // second reduction pass

  udpint ut = (udpint)t;
  ut = (ut << 6) + (spint)(v8 >> 23u);
  v8 &= 0x7fffff;
  ut *= 0x13;
  s = v0 + ((spint)ut & mask);
  c[0] = (spint)(s & mask);
  carry = (s >> 29) + (spint)(ut >> 29);
  c[1] = v1 + carry;
  c[2] = v2;
  c[3] = v3;
  c[4] = v4;
  c[5] = v5;
  c[6] = v6;
  c[7] = v7;
  c[8] = v8;
}

