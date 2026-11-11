#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @scipy_csc_matvec_i64(%arg0: i64, %arg1: i64, %arg2: memref<?xi64>, %arg3: memref<?xi64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi64>
    %1 = bufferization.to_tensor %arg3 : memref<?xi64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg5 : memref<?xf64>
    %4 = bufferization.to_tensor %arg6 : memref<?xf64>
    %5 = arith.index_cast %arg1 : i64 to index
    %6 = affine.for %arg7 = 0 to %5 iter_args(%arg8 = %4) -> (tensor<?xf64>) {
      %extracted = tensor.extract %0[%arg7] : tensor<?xi64>
      %8 = affine.apply #map(%arg7)
      %extracted_0 = tensor.extract %0[%8] : tensor<?xi64>
      %9 = arith.index_cast %extracted_0 : i64 to index
      %10 = arith.index_cast %extracted : i64 to index
      %11 = scf.for %arg9 = %10 to %9 step %c1 iter_args(%arg10 = %arg8) -> (tensor<?xf64>) {
        %extracted_1 = tensor.extract %1[%arg9] : tensor<?xi64>
        %12 = arith.index_cast %extracted_1 : i64 to index
        %extracted_2 = tensor.extract %2[%arg9] : tensor<?xf64>
        %extracted_3 = tensor.extract %3[%arg7] : tensor<?xf64>
        %13 = arith.mulf %extracted_2, %extracted_3 : f64
        %extracted_4 = tensor.extract %arg10[%12] : tensor<?xf64>
        %14 = arith.addf %extracted_4, %13 : f64
        %inserted = tensor.insert %14 into %arg10[%12] : tensor<?xf64>
        scf.yield %inserted : tensor<?xf64>
      }
      affine.yield %11 : tensor<?xf64>
    }
    %7 = bufferization.to_memref %6 : memref<?xf64>
    memref.copy %7, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}

