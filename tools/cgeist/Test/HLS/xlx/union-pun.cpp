// --xlx-headers: a local union written through one member and read through
// another of the same width (the Vitis HLS headers' doubleToRawBits, used by
// every ap_fixed constructor from a double) is that value's bit pattern: the
// single-element struct alloca is scalarised (Lib/XlxProxyScalarize.cc) and
// the read becomes an arith.bitcast, so hls-est sees no memref of an LLVM
// struct. Without --xlx-headers the union stays a struct alloca.
// RUN: cgeist %s --xlx-headers --scalar-wrapper-abi --function='*' -S -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: FileCheck %s --check-prefix=CLEAN < %t.mlir
// RUN: cgeist %s --scalar-wrapper-abi --function='*' -S -o - | FileCheck %s --check-prefix=OFF

// CLEAN-NOT: llvm.struct
// CHECK-LABEL: func.func @pun
// CHECK-NEXT: %[[B:.*]] = arith.bitcast %arg0 : f64 to i64
// CHECK-NEXT: return %[[B]] : i64
// OFF: llvm.struct<(f64)>
extern "C" long long pun(double d) {
  union {
    double d;
    long long i;
  } u;
  u.d = d;
  return u.i;
}

// The same through a member read inside a branch: the read in the scf.if
// region takes the value stored before it.
// CHECK-LABEL: func.func @pun_if
// CHECK: scf.if
// CHECK: arith.bitcast %arg0 : f64 to i64
extern "C" long long pun_if(double d, int c) {
  union {
    double d;
    long long i;
  } u;
  u.d = d;
  long long r = 0;
  if (c)
    r = u.i + 1;
  return r;
}
