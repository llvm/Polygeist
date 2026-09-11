module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @hist(%arg0: i32, %arg1: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = affine.for %arg2 = 0 to %0 iter_args(%arg3 = %c0_i32) -> (i32) {
      %3 = affine.load %arg1[%arg2] : memref<?xf64>
      %4 = arith.cmpf ogt, %3, %cst : f64
      %5 = scf.if %4 -> (i32) {
        %6 = arith.addi %arg3, %c1_i32 : i32
        scf.yield %6 : i32
      } else {
        scf.yield %arg3 : i32
      }
      affine.yield %5 : i32
    }
    %2 = arith.index_cast %1 : i32 to index
    affine.for %arg2 = 0 to %2 {
      %3 = arith.index_cast %arg2 : index to i32
      func.call @use_int(%3) : (i32) -> ()
    }
    return
  }
  func.func private @use_int(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
