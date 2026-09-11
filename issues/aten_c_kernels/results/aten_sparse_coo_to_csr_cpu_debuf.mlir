module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_coo_to_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?xi32>
    %2 = tensor.empty() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %2[] : tensor<i32>
    %3:2 = affine.for %arg2 = 0 to 65 iter_args(%arg3 = %inserted, %arg4 = %0) -> (tensor<i32>, tensor<?xi32>) {
      %extracted = tensor.extract %arg3[] : tensor<i32>
      %5 = arith.index_cast %arg2 : index to i32
      %6 = scf.while (%arg5 = %extracted) : (i32) -> i32 {
        %7 = arith.cmpi slt, %arg5, %c512_i32 : i32
        %8 = arith.index_cast %arg5 : i32 to index
        %extracted_2 = tensor.extract %1[%8] : tensor<?xi32>
        %9 = arith.cmpi slt, %extracted_2, %5 : i32
        %10 = arith.addi %arg5, %c1_i32 : i32
        %11 = arith.select %9, %10, %arg5 : i32
        %12 = arith.select %7, %9, %false : i1
        %13 = arith.select %7, %11, %arg5 : i32
        scf.condition(%12) %13 : i32
      } do {
      ^bb0(%arg5: i32):
        scf.yield %arg5 : i32
      }
      %inserted_0 = tensor.insert %6 into %arg4[%arg2] : tensor<?xi32>
      %inserted_1 = tensor.insert %6 into %arg3[] : tensor<i32>
      affine.yield %inserted_1, %inserted_0 : tensor<i32>, tensor<?xi32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?xi32>
    memref.copy %4, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

