// The Xilinx HLS bit builtins (--xlx-prelude declares them; cgeist lowers
// them to arith, Lib/CGXlxBuiltin.cc, ported from hls-llvm-project ee930469
// CGFPGABuiltin.cpp). On constants every function folds to the value the
// Vitis C simulation (ap_private) gives for the same operation through the
// ap_int API (Test/HLS/realheaders/bits-const.cpp has those).
// No Vitis install and no clang patch needed: the operands are _BitInt.
// RUN: cgeist %s --xlx-prelude --function='*' -S -o - | FileCheck %s
// RUN: cgeist %s --xlx-prelude --function='*' -S -o - | FileCheck %s --check-prefix=NOCALL

typedef unsigned _BitInt(4) u4;
typedef unsigned _BitInt(5) u5;
typedef unsigned _BitInt(7) u7;
typedef unsigned _BitInt(10) u10;
typedef unsigned _BitInt(12) u12;
typedef unsigned _BitInt(16) u16;
typedef unsigned _BitInt(20) u20;
typedef _BitInt(9) s9;

// NOCALL-NOT: call @__builtin_bit
// NOCALL-NOT: call @__fpga
// NOCALL-NOT: llvm.fpga

extern "C" {
// CHECK-LABEL: func.func @sel_fwd()
// CHECK: arith.constant 179 : i32
unsigned sel_fwd() {
  u12 r = 0, v = 0xB3C;
  __builtin_bit_part_select((void *)&r, (void *)&v, 4, 11);
  return (unsigned)r;
}
// Lo > Hi: bits [4, 11] bit-reversed (0xB3 -> 0xCD).
// CHECK-LABEL: func.func @sel_rev()
// CHECK: arith.constant 205 : i32
unsigned sel_rev() {
  u12 r = 0, v = 0xB3C;
  __builtin_bit_part_select((void *)&r, (void *)&v, 11, 4);
  return (unsigned)r;
}
// CHECK-LABEL: func.func @set_fwd()
// CHECK: arith.constant 2908 : i32
unsigned set_fwd() {
  u12 r = 0, v = 0xB3C;
  u4 rep = 5;
  __builtin_bit_part_set((void *)&r, (void *)&v, (void *)&rep, 4, 7);
  return (unsigned)r;
}
// CHECK-LABEL: func.func @set_rev()
// CHECK: arith.constant 128 : i32
unsigned set_rev() {
  u12 r = 0, v = 0;
  u4 rep = 1;
  __builtin_bit_part_set((void *)&r, (void *)&v, (void *)&rep, 7, 4);
  return (unsigned)r;
}
// A replacement wider than the field is cut to it.
// CHECK-LABEL: func.func @set_wide()
// CHECK: arith.constant 4084 : i32
unsigned set_wide() {
  u12 r = 0, v = 0xFFF;
  u16 rep = 0x1234;
  __builtin_bit_part_set((void *)&r, (void *)&v, (void *)&rep, 0, 3);
  return (unsigned)r;
}
// CHECK-LABEL: func.func @bit_select()
// CHECK: arith.constant 6 : i32
int bit_select() {
  s9 v = -2;
  return __builtin_bit_select((void *)&v, 0) +
         2 * __builtin_bit_select((void *)&v, 8) +
         4 * __builtin_bit_select((void *)&v, 1);
}
// CHECK-LABEL: func.func @bit_set()
// CHECK: arith.constant 264 : i32
unsigned bit_set() {
  u12 r = 0, v = 8;
  bool one = true;
  __builtin_bit_set((void *)&r, (void *)&v, (void *)&one, 8);
  return (unsigned)r;
}
// CHECK-LABEL: func.func @concat()
// CHECK: arith.constant 2474 : i32
unsigned concat() {
  u12 r = 0;
  u5 h = 0x13;
  u7 l = 0x2A;
  __builtin_bit_concat((void *)&r, (void *)&h, (void *)&l);
  return (unsigned)r;
}
// and: 1, 0; or: 0, 1; xor: 1, 0 (0x7 has 3 ones, 0x3 has 2)
// CHECK-LABEL: func.func @reduce()
// CHECK: arith.constant 100121 : i32
int reduce() {
  u10 ones = 0x3FF, most = 0x3FE, zero = 0, top = 0x200, three = 0x7,
      two = 0x3;
  return __builtin_bit_and_reduce(&ones) * 100000 +
         __builtin_bit_and_reduce(&most) * 10000 +
         __builtin_bit_or_reduce(&zero) * 1000 +
         __builtin_bit_or_reduce(&top) * 100 +
         __builtin_bit_xor_reduce(&three) * 20 +
         __builtin_bit_xor_reduce(&two) * 10 + 1;
}
// nand(0x3FF) = 0, nor(0) = 1, nxor(0x7) = 0
// CHECK-LABEL: func.func @nreduce()
// CHECK: arith.constant 10 : i32
int nreduce() {
  u10 ones = 0x3FF, zero = 0, three = 0x7;
  return __builtin_bit_nand_reduce(&ones) * 100 +
         __builtin_bit_nor_reduce(&zero) * 10 +
         __builtin_bit_nxor_reduce(&three);
}
// CHECK-LABEL: func.func @clz()
// CHECK: arith.constant 2008 : i32
int clz() {
  u20 v = 0xF00, z = 0;
  return __fpga_ctlz(v) + 100 * __fpga_ctlz(z);
}
// CHECK-LABEL: func.func @from_string()
// CHECK: arith.constant 31123 : i32
unsigned from_string() {
  u16 a = 0, b = 0;
  __builtin_bit_from_string((void *)&a, "0x1F", 10);
  __builtin_bit_from_string((void *)&b, "123", 10);
  return (unsigned)a * 1000 + (unsigned)b;
}

// Runtime indices: both directions are built and selected on lo > hi.
// CHECK-LABEL: func.func @sel_runtime(
// CHECK-DAG: arith.cmpi ugt
// CHECK-NOT: llvm.intr.bitreverse
// CHECK: return
unsigned sel_runtime(u12 v, int lo, int hi) {
  u12 r = 0;
  __builtin_bit_part_select((void *)&r, (void *)&v, lo, hi);
  return (unsigned)r;
}
}
