#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_matmul_maxnnz_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?xi32>
    %2 = bufferization.to_tensor %arg1 : memref<?xi32>
    %3 = bufferization.to_tensor %arg0 : memref<?xi32>
    %4 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %0) -> (tensor<?xi32>) {
      %extracted = tensor.extract %3[%arg5] : tensor<?xi32>
      %6 = affine.apply #map(%arg5)
      %extracted_0 = tensor.extract %3[%6] : tensor<?xi32>
      %7 = arith.index_cast %extracted_0 : i32 to index
      %8 = arith.index_cast %extracted : i32 to index
      %9 = scf.for %arg7 = %8 to %7 step %c1 iter_args(%arg8 = %c0_i32) -> (i32) {
        %extracted_1 = tensor.extract %2[%arg7] : tensor<?xi32>
        %10 = arith.addi %extracted_1, %c1_i32 : i32
        %11 = arith.index_cast %10 : i32 to index
        %extracted_2 = tensor.extract %1[%11] : tensor<?xi32>
        %12 = arith.index_cast %extracted_1 : i32 to index
        %extracted_3 = tensor.extract %1[%12] : tensor<?xi32>
        %13 = arith.subi %extracted_2, %extracted_3 : i32
        %14 = arith.addi %arg8, %13 : i32
        scf.yield %14 : i32
      }
      %inserted = tensor.insert %9 into %arg6[%arg5] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %5 = bufferization.to_memref %4 : memref<?xi32>
    memref.copy %5, %arg4 : memref<?xi32> to memref<?xi32>
    return
  }
}

