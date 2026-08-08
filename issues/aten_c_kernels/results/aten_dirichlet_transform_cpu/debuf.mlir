#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dirichlet_transform_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x16xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %3 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %1) -> (tensor<?x16xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %5 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %5[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %2[%arg2, 0] [1, %c16] [1, 1] : tensor<?x16xf32> to tensor<?xf32>
      %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%inserted : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.addf %out, %in : f32
        linalg.yield %8 : f32
      } -> tensor<f32>
      %extracted = tensor.extract %6[] : tensor<f32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, 0] [1, %c16] [1, 1] : tensor<?x16xf32> to tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %0[%arg2, 0] [1, %c16] [1, 1] : tensor<?x16xf32> to tensor<?xf32>
      %7 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_1 : tensor<?xf32>) outs(%extracted_slice_0 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %8 = arith.divf %in, %extracted : f32
        linalg.yield %8 : f32
      } -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %7 into %arg3[%arg2, 0] [1, %c16] [1, 1] : tensor<?xf32> into tensor<?x16xf32>
      affine.yield %inserted_slice : tensor<?x16xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?x16xf32>
    memref.copy %4, %arg1 : memref<?x16xf32> to memref<?x16xf32>
    return
  }
}

