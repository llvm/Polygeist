// XFAIL: *
// RUN: polygeist-opt --polyhedral-opt %s | FileCheck %s

#set = affine_set<(d0, d1, d2, d3)[s0, s1] : (-d0 - d2 * 8 + s0 - 1 >= 0, -d1 - d3 * 32 + s1 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {

  // upstream mlir does not handle multiplying with symbol as pure affine thus
  // we cannot handle this (can pluto handle it?)
  func.func @indexed_by_symbols(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c31_i32 = arith.constant 31 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.addi %arg0, %c7_i32 : i32
    %4 = arith.divsi %3, %c8_i32 : i32
    %5 = arith.addi %arg1, %c31_i32 : i32
    %6 = arith.divsi %5, %c32_i32 : i32
    %7 = arith.index_cast %4 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    affine.parallel (%arg8, %arg9, %arg10, %arg11) = (0, 0, 0, 0) to (symbol(%7), symbol(%8), 8, 32) {
      affine.if #set(%arg10, %arg11, %arg8, %arg9)[%0, %1] {
        %10 = affine.load %arg5[%arg11 + %arg9 * 32 + (%arg10 + %arg8 * 8) * symbol(%2)] : memref<?xf64>
        %11 = arith.mulf %10, %arg4 : f64
        %12 = affine.for %arg12 = 0 to %2 iter_args(%arg13 = %11) -> (f64) {
          %13 = affine.load %arg6[%arg12 + (%arg10 + %arg8 * 8) * symbol(%2)] : memref<?xf64>
          %14 = arith.mulf %arg3, %13 : f64
          %15 = affine.load %arg7[%arg12 * symbol(%2) + %arg11 + %arg9 * 32] : memref<?xf64>
          %16 = arith.mulf %14, %15 : f64
          %17 = arith.addf %arg13, %16 : f64
          affine.yield %17 : f64
        }
        affine.store %12, %arg5[%arg11 + %arg9 * 32 + (%arg10 + %arg8 * 8) * symbol(%2)] : memref<?xf64>
      }
    }
    return
  }
  // We get a error in scop building if we replace the symbols with consts (1024):
  // [osl] Warning: unexpected number of original iterators (osl_statement_integrity_check).
  func.func @indexed_by_consts(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c31_i32 = arith.constant 31 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.addi %arg0, %c7_i32 : i32
    %4 = arith.divsi %3, %c8_i32 : i32
    %5 = arith.addi %arg1, %c31_i32 : i32
    %6 = arith.divsi %5, %c32_i32 : i32
    %7 = arith.index_cast %4 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    affine.parallel (%arg8, %arg9, %arg10, %arg11) = (0, 0, 0, 0) to (symbol(%7), symbol(%8), 8, 32) {
      affine.if #set(%arg10, %arg11, %arg8, %arg9)[%0, %1] {
        %10 = affine.load %arg5[%arg11 + %arg9 * 32 + (%arg10 + %arg8 * 8) * 1024] : memref<?xf64>
        %11 = arith.mulf %10, %arg4 : f64
        %12 = affine.for %arg12 = 0 to %2 iter_args(%arg13 = %11) -> (f64) {
          %13 = affine.load %arg6[%arg12 + (%arg10 + %arg8 * 8) * 1024] : memref<?xf64>
          %14 = arith.mulf %arg3, %13 : f64
          %15 = affine.load %arg7[%arg12 * 1024 + %arg11 + %arg9 * 32] : memref<?xf64>
          %16 = arith.mulf %14, %15 : f64
          %17 = arith.addf %arg13, %16 : f64
          affine.yield %17 : f64
        }
        affine.store %12, %arg5[%arg11 + %arg9 * 32 + (%arg10 + %arg8 * 8) * 1024] : memref<?xf64>
      }
    }
    return
  }
}
