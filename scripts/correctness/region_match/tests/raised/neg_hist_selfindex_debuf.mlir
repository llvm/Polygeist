#map = affine_map<(d0) -> (d0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @neg_hist_selfindex(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?xi32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3:2 = affine.for %arg3 = 1 to %2 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?xi32>, tensor<?xi32>) {
      %6 = affine.apply #map(%arg3)
      %extracted = tensor.extract %arg5[%6] : tensor<?xi32>
      %7 = arith.index_cast %extracted : i32 to index
      %extracted_0 = tensor.extract %arg4[%7] : tensor<?xi32>
      %8 = arith.addi %extracted_0, %c1_i32 : i32
      %inserted = tensor.insert %8 into %arg4[%7] : tensor<?xi32>
      %9 = affine.apply #map(%arg3)
      %extracted_1 = tensor.extract %arg5[%9] : tensor<?xi32>
      %inserted_2 = tensor.insert %extracted_1 into %arg5[%arg3] : tensor<?xi32>
      affine.yield %inserted, %inserted_2 : tensor<?xi32>, tensor<?xi32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?xi32>
    memref.copy %4, %arg1 : memref<?xi32> to memref<?xi32>
    %5 = bufferization.to_memref %3#0 : memref<?xi32>
    memref.copy %5, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

