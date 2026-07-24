module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_binary_cross_entropy(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.560000e+02 : f32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<?xf32>
    %3 = affine.for %arg3 = 0 to 256 iter_args(%arg4 = %inserted) -> (tensor<?xf32>) {
      %extracted_3 = tensor.extract %1[%arg3] : tensor<?xf32>
      %extracted_4 = tensor.extract %2[%arg3] : tensor<?xf32>
      %6 = func.call @logf(%extracted_4) : (f32) -> f32
      %7 = arith.mulf %extracted_3, %6 : f32
      %extracted_5 = tensor.extract %1[%arg3] : tensor<?xf32>
      %8 = arith.subf %cst_0, %extracted_5 : f32
      %extracted_6 = tensor.extract %2[%arg3] : tensor<?xf32>
      %9 = arith.subf %cst_0, %extracted_6 : f32
      %10 = func.call @logf(%9) : (f32) -> f32
      %11 = arith.mulf %8, %10 : f32
      %12 = arith.addf %7, %11 : f32
      %extracted_7 = tensor.extract %arg4[%c0] : tensor<?xf32>
      %13 = arith.subf %extracted_7, %12 : f32
      %inserted_8 = tensor.insert %13 into %arg4[%c0] : tensor<?xf32>
      affine.yield %inserted_8 : tensor<?xf32>
    }
    %extracted = tensor.extract %3[%c0] : tensor<?xf32>
    %4 = arith.divf %extracted, %cst_1 : f32
    %inserted_2 = tensor.insert %4 into %3[%c0] : tensor<?xf32>
    %5 = bufferization.to_memref %inserted_2 : memref<?xf32>
    memref.copy %5, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @logf(f32) -> f32
}

