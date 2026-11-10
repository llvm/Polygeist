#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @sp_csr_sizeptr(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg2 : memref<?xi32>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %extracted = tensor.extract %0[%c0] : tensor<?xi32>
    %6 = arith.index_cast %extracted : i32 to index
    %7 = affine.for %arg6 = 0 to %6 iter_args(%arg7 = %5) -> (tensor<?xf64>) {
      %extracted_0 = tensor.extract %1[%arg6] : tensor<?xi32>
      %9 = affine.apply #map(%arg6)
      %extracted_1 = tensor.extract %1[%9] : tensor<?xi32>
      %10 = arith.index_cast %extracted_1 : i32 to index
      %11 = arith.index_cast %extracted_0 : i32 to index
      %12 = scf.for %arg8 = %11 to %10 step %c1 iter_args(%arg9 = %cst) -> (f64) {
        %extracted_2 = tensor.extract %3[%arg8] : tensor<?xf64>
        %extracted_3 = tensor.extract %2[%arg8] : tensor<?xi32>
        %13 = arith.index_cast %extracted_3 : i32 to index
        %extracted_4 = tensor.extract %4[%13] : tensor<?xf64>
        %14 = arith.mulf %extracted_2, %extracted_4 : f64
        %15 = arith.addf %arg9, %14 : f64
        scf.yield %15 : f64
      }
      %inserted = tensor.insert %12 into %arg7[%arg6] : tensor<?xf64>
      affine.yield %inserted : tensor<?xf64>
    }
    %8 = bufferization.to_memref %7 : memref<?xf64>
    memref.copy %8, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}

