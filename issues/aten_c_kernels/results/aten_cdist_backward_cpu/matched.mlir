#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cdist_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?x12xf32>, %arg3: memref<?x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x32xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %2 = bufferization.to_tensor %arg3 : memref<?x32xf32>
    %3 = bufferization.to_tensor %arg2 : memref<?x12xf32>
    %4 = bufferization.to_tensor %arg1 : memref<?x32xf32>
    %5 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0] [%c16, %c32] [1, 1] : tensor<?x32xf32> to tensor<?x?xf32>
    %6 = kernel.launch @memset_zero_2D_f32(%extracted_slice) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %6 into %2[0, 0] [%c16, %c32] [1, 1] : tensor<?x?xf32> into tensor<?x32xf32>
    %7 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %inserted_slice) -> (tensor<?x32xf32>) {
      %9 = affine.for %arg6 = 0 to 12 iter_args(%arg7 = %arg5) -> (tensor<?x32xf32>) {
        %alloca = memref.alloca() : memref<f32>
        %10 = bufferization.to_tensor %alloca : memref<f32>
        %inserted = tensor.insert %cst into %10[] : tensor<f32>
        %extracted_slice_0 = tensor.extract_slice %5[%arg4, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
        %extracted_slice_1 = tensor.extract_slice %4[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
        %11 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_0, %extracted_slice_1 : tensor<?xf32>, tensor<?xf32>) outs(%inserted : tensor<f32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %17 = arith.subf %in, %in_7 : f32
          %18 = arith.mulf %17, %17 : f32
          %19 = arith.addf %out, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<f32>
        %extracted = tensor.extract %11[] : tensor<f32>
        %12 = math.sqrt %extracted : f32
        %13 = arith.cmpf oeq, %12, %cst : f32
        %extracted_2 = tensor.extract %3[%arg4, %arg6] : tensor<?x12xf32>
        %14 = arith.divf %extracted_2, %12 : f32
        %15 = arith.select %13, %cst, %14 : f32
        %extracted_slice_3 = tensor.extract_slice %arg7[%arg4, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
        %extracted_slice_4 = tensor.extract_slice %1[%arg4, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
        %extracted_slice_5 = tensor.extract_slice %0[%arg6, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
        %16 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_4, %extracted_slice_5 : tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_3 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %17 = arith.subf %in, %in_7 : f32
          %18 = arith.mulf %15, %17 : f32
          %19 = arith.addf %out, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<?xf32>
        %inserted_slice_6 = tensor.insert_slice %16 into %arg7[%arg4, 0] [1, %c32] [1, 1] : tensor<?xf32> into tensor<?x32xf32>
        affine.yield %inserted_slice_6 : tensor<?x32xf32>
      }
      affine.yield %9 : tensor<?x32xf32>
    }
    %8 = bufferization.to_memref %7 : memref<?x32xf32>
    memref.copy %8, %arg3 : memref<?x32xf32> to memref<?x32xf32>
    return
  }
}

