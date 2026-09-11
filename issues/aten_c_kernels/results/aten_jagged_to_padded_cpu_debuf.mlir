#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_jagged_to_padded_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %0) -> (tensor<?x64xf32>) {
      %5 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %arg4) -> (tensor<?x64xf32>) {
        %6 = arith.index_cast %arg5 : index to i32
        %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
        %7 = arith.addi %extracted, %6 : i32
        %8 = affine.apply #map(%arg3)
        %extracted_0 = tensor.extract %1[%8] : tensor<?xi32>
        %9 = arith.cmpi slt, %7, %extracted_0 : i32
        %10 = arith.index_cast %7 : i32 to index
        %extracted_1 = tensor.extract %2[%10] : tensor<?xf32>
        %11 = arith.select %9, %extracted_1, %cst : f32
        %inserted = tensor.insert %11 into %arg6[%arg3, %arg5] : tensor<?x64xf32>
        affine.yield %inserted : tensor<?x64xf32>
      }
      affine.yield %5 : tensor<?x64xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?x64xf32>
    memref.copy %4, %arg2 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

