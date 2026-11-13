#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @scipy_csc_matvec_beta0(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xi32>
    %4 = bufferization.to_tensor %arg2 : memref<?xi32>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = polygeist.submap(%0, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%6 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    %8 = polygeist.submapInverse(%0, %7, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %9 = arith.index_cast %arg1 : i32 to index
    %10 = affine.for %arg7 = 0 to %9 iter_args(%arg8 = %8) -> (tensor<?xf64>) {
      %extracted = tensor.extract %4[%arg7] : tensor<?xi32>
      %12 = affine.apply #map1(%arg7)
      %extracted_0 = tensor.extract %4[%12] : tensor<?xi32>
      %13 = arith.index_cast %extracted_0 : i32 to index
      %14 = arith.index_cast %extracted : i32 to index
      %15 = scf.for %arg9 = %14 to %13 step %c1 iter_args(%arg10 = %arg8) -> (tensor<?xf64>) {
        %extracted_1 = tensor.extract %3[%arg9] : tensor<?xi32>
        %16 = arith.index_cast %extracted_1 : i32 to index
        %extracted_2 = tensor.extract %2[%arg9] : tensor<?xf64>
        %extracted_3 = tensor.extract %1[%arg7] : tensor<?xf64>
        %17 = arith.mulf %extracted_2, %extracted_3 : f64
        %extracted_4 = tensor.extract %arg10[%16] : tensor<?xf64>
        %18 = arith.addf %extracted_4, %17 : f64
        %inserted = tensor.insert %18 into %arg10[%16] : tensor<?xf64>
        scf.yield %inserted : tensor<?xf64>
      }
      affine.yield %15 : tensor<?xf64>
    }
    %11 = bufferization.to_memref %10 : memref<?xf64>
    memref.copy %11, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}

