#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_addmv_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?xi32>
    %5 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %4[%arg5] : tensor<?xi32>
      %7 = affine.apply #map(%arg5)
      %extracted_0 = tensor.extract %4[%7] : tensor<?xi32>
      %8 = arith.index_cast %extracted_0 : i32 to index
      %9 = arith.index_cast %extracted : i32 to index
      %10 = scf.for %arg7 = %9 to %8 step %c1 iter_args(%arg8 = %cst) -> (f32) {
        %extracted_1 = tensor.extract %2[%arg7] : tensor<?xf32>
        %extracted_2 = tensor.extract %3[%arg7] : tensor<?xi32>
        %11 = arith.index_cast %extracted_2 : i32 to index
        %extracted_3 = tensor.extract %1[%11] : tensor<?xf32>
        %12 = arith.mulf %extracted_1, %extracted_3 : f32
        %13 = arith.addf %arg8, %12 : f32
        scf.yield %13 : f32
      }
      %inserted = tensor.insert %10 into %arg6[%arg5] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg4 : memref<?xf32> to memref<?xf32>
    return
  }
}

