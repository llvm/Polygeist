#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @renamed_kernel(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %inserted = tensor.insert %c0_i32 into %1[%c0] : tensor<?xi32>
    %2 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %inserted) -> (tensor<?xi32>) {
      %extracted = tensor.extract %arg3[%arg2] : tensor<?xi32>
      %extracted_0 = tensor.extract %0[%arg2] : tensor<?xi32>
      %4 = arith.addi %extracted, %extracted_0 : i32
      %5 = affine.apply #map(%arg2)
      %inserted_1 = tensor.insert %4 into %arg3[%5] : tensor<?xi32>
      affine.yield %inserted_1 : tensor<?xi32>
    }
    %3 = bufferization.to_memref %2 : memref<?xi32>
    memref.copy %3, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

