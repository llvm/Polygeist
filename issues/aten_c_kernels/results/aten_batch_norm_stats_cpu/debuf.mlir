#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_stats_cpu(%arg0: memref<?x16x16x16xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.048000e+03 : f32
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x16x16x16xf32>
    %3:2 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?xf32>, tensor<?xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %6 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %6[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %2[0, %arg3, 0, 0] [%c8, 1, %c16, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?x?x?xf32>
      %7 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "reduction"], library_call = ""} ins(%extracted_slice : tensor<?x?x?xf32>) outs(%inserted : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.addf %out, %in : f32
        linalg.yield %12 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %7[] : tensor<f32>
      %8 = arith.divf %extracted, %cst_0 : f32
      %alloca_1 = memref.alloca() : memref<f32>
      %9 = bufferization.to_tensor %alloca_1 : memref<f32>
      %inserted_2 = tensor.insert %cst into %9[] : tensor<f32>
      %extracted_slice_3 = tensor.extract_slice %2[0, %arg3, 0, 0] [%c8, 1, %c16, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?x?x?xf32>
      %10 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "reduction"], library_call = ""} ins(%extracted_slice_3 : tensor<?x?x?xf32>) outs(%inserted_2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.subf %in, %8 : f32
        %13 = arith.mulf %12, %12 : f32
        %14 = arith.addf %out, %13 : f32
        linalg.yield %14 : f32
      } -> tensor<f32>
      %extracted_4 = tensor.extract %10[] : tensor<f32>
      %inserted_5 = tensor.insert %8 into %arg4[%arg3] : tensor<?xf32>
      %11 = arith.divf %extracted_4, %cst_0 : f32
      %inserted_6 = tensor.insert %11 into %arg5[%arg3] : tensor<?xf32>
      affine.yield %inserted_5, %inserted_6 : tensor<?xf32>, tensor<?xf32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?xf32>
    memref.copy %4, %arg2 : memref<?xf32> to memref<?xf32>
    %5 = bufferization.to_memref %3#0 : memref<?xf32>
    memref.copy %5, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

