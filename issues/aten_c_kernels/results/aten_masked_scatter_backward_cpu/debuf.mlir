module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_masked_scatter_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = tensor.empty() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %3[] : tensor<i32>
    %4:2 = affine.for %arg3 = 0 to 512 iter_args(%arg4 = %inserted, %arg5 = %0) -> (tensor<i32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg4[] : tensor<i32>
      %extracted_0 = tensor.extract %1[%arg3] : tensor<?xi32>
      %6 = arith.cmpi ne, %extracted_0, %c0_i32 : i32
      %7:2 = scf.if %6 -> (i32, tensor<?xf32>) {
        %8 = arith.addi %extracted, %c1_i32 : i32
        %9 = arith.index_cast %extracted : i32 to index
        %extracted_2 = tensor.extract %2[%arg3] : tensor<?xf32>
        %inserted_3 = tensor.insert %extracted_2 into %arg5[%9] : tensor<?xf32>
        scf.yield %8, %inserted_3 : i32, tensor<?xf32>
      } else {
        scf.yield %extracted, %arg5 : i32, tensor<?xf32>
      }
      %inserted_1 = tensor.insert %7#0 into %arg4[] : tensor<i32>
      affine.yield %inserted_1, %7#1 : tensor<i32>, tensor<?xf32>
    }
    %5 = bufferization.to_memref %4#1 : memref<?xf32>
    memref.copy %5, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

