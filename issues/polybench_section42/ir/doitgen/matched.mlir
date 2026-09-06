#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg0 : i32 to index
    %6:2 = affine.for %arg6 = 0 to %5 iter_args(%arg7 = %2, %arg8 = %0) -> (tensor<?xf64>, tensor<?x?x?xf64>) {
      %9:2 = affine.for %arg9 = 0 to %3 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (tensor<?xf64>, tensor<?x?x?xf64>) {
        %10 = kernel.launch @memset_zero_1D(%arg10) : (tensor<?xf64>) -> tensor<?xf64>
        %extracted_slice = tensor.extract_slice %arg11[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?xf64>
        %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%4, %4] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
        %extracted_slice_1 = tensor.extract_slice %10[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
        %11 = kernel.launch @cublasDgemv_T(%extracted_slice_0, %extracted_slice, %extracted_slice_1) : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
        %inserted_slice = tensor.insert_slice %11 into %10[0] [%4] [1] : tensor<?xf64> into tensor<?xf64>
        %extracted_slice_2 = tensor.extract_slice %arg11[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : tensor<?x?x?xf64> to tensor<?xf64>
        %12 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%11 : tensor<?xf64>) outs(%extracted_slice_2 : tensor<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          linalg.yield %in : f64
        } -> tensor<?xf64>
        %inserted_slice_3 = tensor.insert_slice %12 into %arg11[%arg6, %arg9, 0] [1, 1, %4] [1, 1, 1] : tensor<?xf64> into tensor<?x?x?xf64>
        affine.yield %inserted_slice, %inserted_slice_3 : tensor<?xf64>, tensor<?x?x?xf64>
      }
      affine.yield %9#0, %9#1 : tensor<?xf64>, tensor<?x?x?xf64>
    }
    %7 = bufferization.to_memref %6#1 : memref<?x?x?xf64>
    memref.copy %7, %arg3 : memref<?x?x?xf64> to memref<?x?x?xf64>
    %8 = bufferization.to_memref %6#0 : memref<?xf64>
    memref.copy %8, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}

