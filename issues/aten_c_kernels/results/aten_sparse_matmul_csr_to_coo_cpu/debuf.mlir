#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_matmul_csr_to_coo_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?xi32>
    %2 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %0) -> (tensor<?xi32>) {
      %4 = arith.index_cast %arg2 : index to i32
      %extracted = tensor.extract %1[%arg2] : tensor<?xi32>
      %5:2 = scf.while (%arg4 = %extracted, %arg5 = %arg3) : (i32, tensor<?xi32>) -> (i32, tensor<?xi32>) {
        %6 = affine.apply #map(%arg2)
        %extracted_0 = tensor.extract %1[%6] : tensor<?xi32>
        %7 = arith.cmpi slt, %arg4, %extracted_0 : i32
        scf.condition(%7) %arg4, %arg5 : i32, tensor<?xi32>
      } do {
      ^bb0(%arg4: i32, %arg5: tensor<?xi32>):
        %6 = arith.index_cast %arg4 : i32 to index
        %inserted = tensor.insert %4 into %arg5[%6] : tensor<?xi32>
        %7 = arith.addi %arg4, %c1_i32 : i32
        scf.yield %7, %inserted : i32, tensor<?xi32>
      }
      affine.yield %5#1 : tensor<?xi32>
    }
    %3 = bufferization.to_memref %2 : memref<?xi32>
    memref.copy %3, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

