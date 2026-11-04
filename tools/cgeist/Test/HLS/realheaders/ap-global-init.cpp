// Globals of ap types initialised from numbers (OPEN_ISSUES O63): the real
// headers' ap_uint / ap_fixed constructors are not constexpr, so clang has no
// constant for the initialiser; cgeist folds the construction from an integer
// or floating-point constant with the class's quantisation and overflow modes
// (XlxStringBits.cc foldNumericConstruct), as the Vitis front end folds the
// global's constructor. The values are those of the C simulation of the same
// headers (g++ + Vitis/include).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
#include <ap_int.h>

// CHECK-DAG: memref.global "private" @_ZL1K : memref<1xi8> = dense<-1>
const static ap_uint<8> K = -1;
// CHECK-DAG: memref.global "private" @_ZL2K3 : memref<1xi8> = dense<7>
static ap_uint<8> K3 = 7;
// CHECK-DAG: memref.global "private" @_ZL1T : memref<3xi8> = dense<[1, 2, 3]>
const ap_uint<8> T[3] = {1, 2, 3};
// 0.5 and -1.25 at 12 fractional bits.
// CHECK-DAG: memref.global "private" @_ZL1F : memref<2xi16> = dense<[2048, -5120]>
const ap_fixed<16, 4> F[2] = {0.5, -1.25};
// AP_TRN floors, AP_RND rounds half up: 0.3 * 2^4 = 4.8, -0.3 * 2^4 = -4.8.
// CHECK-DAG: memref.global "private" @_ZL2FT : memref<2xi8> = dense<[4, -5]>
const ap_fixed<8, 4> FT[2] = {0.3, -0.3};
// CHECK-DAG: memref.global "private" @_ZL2FR : memref<2xi8> = dense<[5, -5]>
const ap_fixed<8, 4, AP_RND> FR[2] = {0.3, -0.3};
// AP_WRAP wraps 9.5 into [-8, 8); AP_SAT clamps it to 127 / 16.
// CHECK-DAG: memref.global "private" @_ZL2OW : memref<1xi8> = dense<-104>
const ap_fixed<8, 4> OW = 9.5;
// CHECK-DAG: memref.global "private" @_ZL2OS : memref<1xi8> = dense<127>
const ap_fixed<8, 4, AP_TRN, AP_SAT> OS = 9.5;
// An unsigned fixed-point value from an int, and an ap_int from a bool.
// CHECK-DAG: memref.global "private" @_ZL2UF : memref<1xi6> = dense<12>
const ap_ufixed<6, 4> UF = 3;
// CHECK-DAG: memref.global "private" @_ZL2B1 : memref<1xi3> = dense<1>
const ap_int<3> B1 = true;
// Default-constructed ap objects in static storage are zero (the headers'
// default constructors are `= default` and set nothing; the object is
// zero-initialised first), with or without a `{0}` initialiser.
// CHECK-DAG: memref.global @Z : memref<4xi16> = dense<0>
ap_fixed<16, 4> Z[4];
// CHECK-DAG: memref.global @Z0 : memref<3xi10> = dense<[5, 0, 0]>
ap_uint<10> Z0[3] = {5};
// An array of structs with ap members (review defect 4): memref.global
// cannot hold llvm.struct elements, so the constant is one attribute per
// scalar in memory order, which hls-est's --split-struct-memrefs turns into
// the members' initial values. 0.5 / -1.25 / 127.99 / -128.5 at 8
// fractional bits (AP_TRN, AP_WRAP: -128.5 wraps to 32640 - 65536); 1, -2,
// 31 and -33 (wraps to 31) in 6 bits.
struct C { ap_fixed<16, 8> a; ap_int<6> b; };
// CHECK-DAG: memref.global "private" @_ZL4ctab : memref<4x!llvm.struct<(i16, i6)>> = uninitialized {hls.struct_initial_value = [128 : i16, 1 : i6, -320 : i16, -2 : i6, 32765 : i16, 31 : i6, 32640 : i16, 31 : i6]}
static const C ctab[4] = {{0.5, 1}, {-1.25, -2}, {127.99, 31}, {-128.5, -33}};

void top(ap_uint<8> o[8], ap_fixed<16, 4> f[8], ap_fixed<8, 4> g[8],
         ap_ufixed<6, 4> *u, ap_int<3> *b, ap_uint<2> k, ap_int<32> *c) {
  *c = ctab[k].a.range(15, 0) + (ctab[k].b << 16) + Z[k] + Z0[k];
  o[0] = K;
  o[1] = K3;
  o[2] = T[2];
  f[0] = F[0];
  f[1] = F[1];
  g[0] = FT[0];
  g[1] = FT[1];
  g[2] = FR[0];
  g[3] = FR[1];
  g[4] = OW;
  g[5] = OS;
  *u = UF;
  *b = B1;
}
