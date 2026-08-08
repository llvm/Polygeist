#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_padded_to_jagged_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %3 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %0) -> (tensor<?xf32>) {
      %5:2 = scf.while (%arg5 = %c0_i32, %arg6 = %arg4) : (i32, tensor<?xf32>) -> (i32, tensor<?xf32>) {
        %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
        %6 = arith.addi %extracted, %arg5 : i32
        %7 = affine.apply #map(%arg3)
        %extracted_0 = tensor.extract %1[%7] : tensor<?xi32>
        %8 = arith.cmpi slt, %6, %extracted_0 : i32
        scf.condition(%8) %arg5, %arg6 : i32, tensor<?xf32>
      } do {
      ^bb0(%arg5: i32, %arg6: tensor<?xf32>):
        %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
        %6 = arith.addi %extracted, %arg5 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = arith.index_cast %arg5 : i32 to index
        %extracted_0 = tensor.extract %2[%arg3, %8] : tensor<?x64xf32>
        %inserted = tensor.insert %extracted_0 into %arg6[%7] : tensor<?xf32>
        %9 = arith.addi %arg5, %c1_i32 : i32
        scf.yield %9, %inserted : i32, tensor<?xf32>
      }
      affine.yield %5#1 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

