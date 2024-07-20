// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal $ISL_OPT_PLACEHOLDER %s 2>&1 | FileCheck %s
// CHECK-NOT: isl_ctx not freed
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @f(%arg0: i64, %arg1: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i64 to index
    affine.for %arg2 = 1 to %0 {
      affine.for %arg3 = 0 to %0 {
        %1 = affine.load %arg1[%arg2, %arg3] : memref<?x?xf64>
        %2 = affine.load %arg1[%arg2 - 1, %arg3] : memref<?x?xf64>
        %3 = arith.addf %1, %2 : f64
        affine.store %3, %arg1[%arg2, %arg3] : memref<?x?xf64>
      }
    }
    return
  }
}
// RUN: mkdir -p %t/schedules
// RUN: mkdir -p %t/accesses
// RUN:  polygeist-opt --polyhedral-opt --use-polyhedral-optimizer=islexternal --islexternal-dump-schedules=%t/schedules --islexternal-dump-accesses=%t/accesses $ISL_OPT_PLACEHOLDER %s && find %t/schedules/ %t/accesses/ -type f -print0 | sort -z | xargs -0r cat | FileCheck --check-prefix=ISL_OUT %s
// ISL_OUT: domain: "[P0] -> { S0[i0, i1] : 0 < i0 < P0 and 0 <= i1 < P0 }"
// ISL_OUT: accesses:
// ISL_OUT:   - S0:
// ISL_OUT:       reads:
// ISL_OUT:         - "[P0] -> { [i0, i1] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
// ISL_OUT:         - "[P0] -> { [i0, i1] -> A1[o0, o1] : o0 = -1 + i0 and o1 = i1 }"
// ISL_OUT:       writes:
// ISL_OUT:         - "[P0] -> { [i0, i1] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
// ISL_OUT: { domain: "[P0] -> { S0[i0, i1] : 0 < i0 < P0 and 0 <= i1 < P0 }", child: { schedule: "[P0] -> L1[{ S0[i0, i1] -> [(i0)] }]", child: { schedule: "[P0] -> L0[{ S0[i0, i1] -> [(i1)] }]" } } }
