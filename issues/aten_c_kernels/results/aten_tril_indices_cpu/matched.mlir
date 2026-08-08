#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_tril_indices_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?xi32>
    %2:2 = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %1, %arg4 = %0) -> (tensor<?xi32>, tensor<?xi32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.addi %5, %c1_i32 : i32
      %7 = arith.muli %5, %6 : i32
      %8 = arith.divsi %7, %c2_i32 : i32
      %9:2 = affine.for %arg5 = 0 to #map(%arg2) iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<?xi32>, tensor<?xi32>) {
        %10 = arith.index_cast %arg5 : index to i32
        %11 = arith.addi %8, %10 : i32
        %12 = arith.index_cast %11 : i32 to index
        %inserted = tensor.insert %5 into %arg6[%12] : tensor<?xi32>
        %inserted_0 = tensor.insert %10 into %arg7[%12] : tensor<?xi32>
        affine.yield %inserted, %inserted_0 : tensor<?xi32>, tensor<?xi32>
      }
      affine.yield %9#0, %9#1 : tensor<?xi32>, tensor<?xi32>
    }
    %3 = bufferization.to_memref %2#1 : memref<?xi32>
    memref.copy %3, %arg1 : memref<?xi32> to memref<?xi32>
    %4 = bufferization.to_memref %2#0 : memref<?xi32>
    memref.copy %4, %arg0 : memref<?xi32> to memref<?xi32>
    return
  }
}

