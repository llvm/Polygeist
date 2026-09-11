module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_full_coo_indices_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?xi32>
    %2 = tensor.empty() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %2[] : tensor<i32>
    %3:3 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %inserted, %arg4 = %1, %arg5 = %0) -> (tensor<i32>, tensor<?xi32>, tensor<?xi32>) {
      %extracted = tensor.extract %arg3[] : tensor<i32>
      %6 = arith.index_cast %arg2 : index to i32
      %7 = arith.index_cast %extracted : i32 to index
      %8 = arith.addi %7, %c32 : index
      %9 = arith.index_cast %8 : index to i32
      %10:2 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?xi32>, tensor<?xi32>) {
        %11 = arith.addi %7, %arg6 : index
        %12 = arith.index_cast %arg6 : index to i32
        %inserted_1 = tensor.insert %6 into %arg7[%11] : tensor<?xi32>
        %inserted_2 = tensor.insert %12 into %arg8[%11] : tensor<?xi32>
        affine.yield %inserted_1, %inserted_2 : tensor<?xi32>, tensor<?xi32>
      }
      %inserted_0 = tensor.insert %9 into %arg3[] : tensor<i32>
      affine.yield %inserted_0, %10#0, %10#1 : tensor<i32>, tensor<?xi32>, tensor<?xi32>
    }
    %4 = bufferization.to_memref %3#2 : memref<?xi32>
    memref.copy %4, %arg1 : memref<?xi32> to memref<?xi32>
    %5 = bufferization.to_memref %3#1 : memref<?xi32>
    memref.copy %5, %arg0 : memref<?xi32> to memref<?xi32>
    return
  }
}

