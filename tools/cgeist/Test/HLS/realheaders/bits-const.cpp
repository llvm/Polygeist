// Bit and range operations of the real Vitis HLS headers on constants: each
// function folds to the value the Vitis C simulation (ap_private, g++ against
// Vitis/include) prints for the same code. The ops go through the headers'
// __builtin_bit_* calls (lowered by cgeist, --xlx-prelude) and the
// ap_range_ref / ap_bit_ref / ap_concat_ref proxies (--xlx-headers).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_int.h>

extern "C" {
// CHECK-LABEL: func.func @c_range()
// CHECK-NEXT: arith.constant 179 : i32
// CHECK-NEXT: return
int c_range() { ap_uint<12> x = 0xB3C; return x.range(11, 4).to_int(); }
// Lo > Hi: the field reversed (0xB3 -> 0xCD).
// CHECK-LABEL: func.func @c_range_rev()
// CHECK-NEXT: arith.constant 205 : i32
// CHECK-NEXT: return
int c_range_rev() { ap_uint<12> x = 0xB3C; return x.range(4, 11).to_int(); }
// CHECK-LABEL: func.func @c_range_op()
// CHECK-NEXT: arith.constant 60 : i32
// CHECK-NEXT: return
int c_range_op() { ap_uint<12> x = 0xB3C; return x(7, 0).to_int(); }
// CHECK-LABEL: func.func @c_setrange()
// CHECK-NEXT: arith.constant 2908 : i32
// CHECK-NEXT: return
int c_setrange() { ap_uint<12> x = 0xB3C; x.range(7, 4) = 0x5; return x.to_int(); }
// CHECK-LABEL: func.func @c_setrange_rev()
// CHECK-NEXT: arith.constant 128 : i32
// CHECK-NEXT: return
int c_setrange_rev() { ap_uint<12> x = 0; x.range(4, 7) = 0x1; return x.to_int(); }
// CHECK-LABEL: func.func @c_setrange_wide()
// CHECK-NEXT: arith.constant 4084 : i32
// CHECK-NEXT: return
int c_setrange_wide() { ap_uint<12> x = 0xFFF; x.range(3, 0) = 0x1234; return x.to_int(); }
// CHECK-LABEL: func.func @c_bit()
// CHECK-NEXT: arith.constant 6 : i32
// CHECK-NEXT: return
int c_bit() { ap_int<9> x = -2; return (x[0] ? 1 : 0) + (x[8] ? 2 : 0) + (x[1] ? 4 : 0); }
// CHECK-LABEL: func.func @c_setbit()
// CHECK-NEXT: arith.constant -248 : i32
// CHECK-NEXT: return
int c_setbit() { ap_int<9> x = 0; x[8] = 1; x[3] = 1; return x.to_int(); }
// CHECK-LABEL: func.func @c_concat()
// CHECK-NEXT: arith.constant 2474 : i32
// CHECK-NEXT: return
int c_concat() { ap_uint<5> a = 0x13; ap_uint<7> b = 0x2A; ap_uint<12> r = (a, b); return r.to_int(); }
// CHECK-LABEL: func.func @c_concat_m()
// CHECK-NEXT: arith.constant 217 : i32
// CHECK-NEXT: return
int c_concat_m() { ap_int<4> a = -3; ap_uint<4> b = 9; ap_uint<8> r = a.concat(b); return r.to_int(); }
// CHECK-LABEL: func.func @c_and_reduce()
// CHECK-NEXT: arith.constant 2 : i32
// CHECK-NEXT: return
int c_and_reduce() { ap_uint<10> a = 0x3FF, b = 0x3FE; return a.and_reduce() * 2 + b.and_reduce(); }
// CHECK-LABEL: func.func @c_or_reduce()
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: return
int c_or_reduce() { ap_uint<10> a = 0, b = 0x200; return a.or_reduce() * 2 + b.or_reduce(); }
// CHECK-LABEL: func.func @c_xor_reduce()
// CHECK-NEXT: arith.constant 2 : i32
// CHECK-NEXT: return
int c_xor_reduce() { ap_uint<10> a = 0x7, b = 0x3; return a.xor_reduce() * 2 + b.xor_reduce(); }
// CHECK-LABEL: func.func @c_nand_reduce()
// CHECK-NEXT: arith.constant 0 : i32
// CHECK-NEXT: return
int c_nand_reduce() { ap_uint<10> a = 0x3FF; return a.nand_reduce(); }
// CHECK-LABEL: func.func @c_nor_reduce()
// CHECK-NEXT: arith.constant 1 : i32
// CHECK-NEXT: return
int c_nor_reduce() { ap_uint<10> a = 0; return a.nor_reduce(); }
// CHECK-LABEL: func.func @c_xnor_reduce()
// CHECK-NEXT: arith.constant 0 : i32
// CHECK-NEXT: return
int c_xnor_reduce() { ap_uint<10> a = 0x7; return a.xnor_reduce(); }
// CHECK-LABEL: func.func @c_clz()
// CHECK-NEXT: arith.constant 8 : i32
// CHECK-NEXT: return
int c_clz() { ap_uint<20> a = 0x00F00; return a.countLeadingZeros(); }
// CHECK-LABEL: func.func @c_clz0()
// CHECK-NEXT: arith.constant 20 : i32
// CHECK-NEXT: return
int c_clz0() { ap_uint<20> a = 0; return a.countLeadingZeros(); }
// CHECK-LABEL: func.func @c_wide_range()
// CHECK-NEXT: arith.constant 4886718345 : i64
// CHECK-NEXT: return
long long c_wide_range() {
  ap_uint<100> x = 0x123456789ULL;
  x = (x << 60) | ap_uint<100>(0xABCDEULL);
  ap_uint<40> r = x.range(99, 60);
  return (long long)r.to_uint64();
}
}
