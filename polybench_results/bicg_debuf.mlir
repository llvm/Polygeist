#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<?x38xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg6 : memref<?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg2 : memref<?x38xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = polygeist.submap(%3, %5) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%6 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?xf64>
    %8 = polygeist.submapInverse(%3, %7, %5) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %9 = arith.index_cast %arg1 : i32 to index
    %10:2 = affine.for %arg7 = 0 to %9 iter_args(%arg8 = %8, %arg9 = %2) -> (tensor<?xf64>, tensor<?xf64>) {
      %inserted = tensor.insert %cst into %arg9[%arg7] : tensor<?xf64>
      %13:2 = affine.for %arg10 = 0 to %5 iter_args(%arg11 = %arg8, %arg12 = %inserted) -> (tensor<?xf64>, tensor<?xf64>) {
        %extracted = tensor.extract %arg11[%arg10] : tensor<?xf64>
        %extracted_0 = tensor.extract %0[%arg7] : tensor<?xf64>
        %extracted_1 = tensor.extract %4[%arg7, %arg10] : tensor<?x38xf64>
        %14 = arith.mulf %extracted_0, %extracted_1 : f64
        %15 = arith.addf %extracted, %14 : f64
        %inserted_2 = tensor.insert %15 into %arg11[%arg10] : tensor<?xf64>
        %extracted_3 = tensor.extract %arg12[%arg7] : tensor<?xf64>
        %extracted_4 = tensor.extract %4[%arg7, %arg10] : tensor<?x38xf64>
        %extracted_5 = tensor.extract %1[%arg10] : tensor<?xf64>
        %16 = arith.mulf %extracted_4, %extracted_5 : f64
        %17 = arith.addf %extracted_3, %16 : f64
        %inserted_6 = tensor.insert %17 into %arg12[%arg7] : tensor<?xf64>
        affine.yield %inserted_2, %inserted_6 : tensor<?xf64>, tensor<?xf64>
      }
      affine.yield %13#0, %13#1 : tensor<?xf64>, tensor<?xf64>
    }
    %11 = bufferization.to_memref %10#1 : memref<?xf64>
    memref.copy %11, %arg4 : memref<?xf64> to memref<?xf64>
    %12 = bufferization.to_memref %10#0 : memref<?xf64>
    memref.copy %12, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}

