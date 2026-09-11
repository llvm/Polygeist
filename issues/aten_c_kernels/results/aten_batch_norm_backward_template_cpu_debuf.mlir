#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_backward_template_cpu(%arg0: memref<?x16x16x16xf32>, %arg1: memref<?x16x16x16xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x16x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.048000e+03 : f32
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?x16x16x16xf32>
    %3 = bufferization.to_tensor %arg0 : memref<?x16x16x16xf32>
    %4 = bufferization.to_tensor %arg4 : memref<?x16x16x16xf32>
    %5 = bufferization.to_tensor %arg2 : memref<?xf32>
    %6 = bufferization.to_tensor %arg1 : memref<?x16x16x16xf32>
    %7 = bufferization.to_tensor %arg0 : memref<?x16x16x16xf32>
    %alloca = memref.alloca(%c16) : memref<?xf32>
    %alloca_1 = memref.alloca(%c16) : memref<?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg5 = 0 to 16 {
      %alloca_8 = memref.alloca(%c8) : memref<?xf32>
      %10 = bufferization.to_tensor %alloca_8 : memref<?xf32>
      %alloca_9 = memref.alloca(%c8) : memref<?xf32>
      %11 = bufferization.to_tensor %alloca_9 : memref<?xf32>
      %12:2 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %10, %arg8 = %11) -> (tensor<?xf32>, tensor<?xf32>) {
        %13 = affine.load %alloca[%arg5] : memref<?xf32>
        %14 = affine.load %alloca_1[%arg5] : memref<?xf32>
        %extracted = tensor.extract %5[%arg5] : tensor<?xf32>
        %inserted = tensor.insert %13 into %arg7[%arg6] : tensor<?xf32>
        %inserted_10 = tensor.insert %14 into %arg8[%arg6] : tensor<?xf32>
        %alloca_11 = memref.alloca(%c16) : memref<?xf32>
        %15 = bufferization.to_tensor %alloca_11 : memref<?xf32>
        %alloca_12 = memref.alloca(%c16) : memref<?xf32>
        %16 = bufferization.to_tensor %alloca_12 : memref<?xf32>
        %17:4 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %inserted, %arg11 = %inserted_10, %arg12 = %15, %arg13 = %16) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
          %extracted_15 = tensor.extract %arg10[%arg6] : tensor<?xf32>
          %extracted_16 = tensor.extract %arg11[%arg6] : tensor<?xf32>
          %inserted_17 = tensor.insert %extracted_15 into %arg12[%arg9] : tensor<?xf32>
          %inserted_18 = tensor.insert %extracted_16 into %arg13[%arg9] : tensor<?xf32>
          %extracted_slice_19 = tensor.extract_slice %inserted_17[%arg9] [1] [1] : tensor<?xf32> to tensor<f32>
          %extracted_slice_20 = tensor.extract_slice %inserted_18[%arg9] [1] [1] : tensor<?xf32> to tensor<f32>
          %extracted_slice_21 = tensor.extract_slice %7[%arg6, %arg5, %arg9, 0] [1, 1, 1, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?xf32>
          %extracted_slice_22 = tensor.extract_slice %6[%arg6, %arg5, %arg9, 0] [1, 1, 1, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?xf32>
          %18:2 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_21, %extracted_slice_22 : tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_19, %extracted_slice_20 : tensor<f32>, tensor<f32>) {
          ^bb0(%in: f32, %in_29: f32, %out: f32, %out_30: f32):
            %19 = arith.addf %out_30, %in : f32
            %20 = arith.subf %in_29, %extracted : f32
            %21 = arith.mulf %in, %20 : f32
            %22 = arith.addf %out, %21 : f32
            linalg.yield %22, %19 : f32, f32
          } -> (tensor<f32>, tensor<f32>)
          %inserted_slice_23 = tensor.insert_slice %18#1 into %inserted_18[%arg9] [1] [1] : tensor<f32> into tensor<?xf32>
          %inserted_slice_24 = tensor.insert_slice %18#0 into %inserted_17[%arg9] [1] [1] : tensor<f32> into tensor<?xf32>
          %extracted_25 = tensor.extract %inserted_slice_24[%arg9] : tensor<?xf32>
          %extracted_26 = tensor.extract %inserted_slice_23[%arg9] : tensor<?xf32>
          %inserted_27 = tensor.insert %extracted_25 into %arg10[%arg6] : tensor<?xf32>
          %inserted_28 = tensor.insert %extracted_26 into %arg11[%arg6] : tensor<?xf32>
          affine.yield %inserted_27, %inserted_28, %inserted_slice_24, %inserted_slice_23 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
        }
        %extracted_13 = tensor.extract %17#0[%arg6] : tensor<?xf32>
        %extracted_14 = tensor.extract %17#1[%arg6] : tensor<?xf32>
        affine.store %extracted_13, %alloca[%arg5] : memref<?xf32>
        affine.store %extracted_14, %alloca_1[%arg5] : memref<?xf32>
        affine.yield %17#0, %17#1 : tensor<?xf32>, tensor<?xf32>
      }
    } {polygeist.was_parallel}
    %reinterpret_cast = memref.reinterpret_cast %alloca to offset: [0], sizes: [%c16], strides: [1] : memref<?xf32> to memref<?xf32>
    %subview = memref.subview %reinterpret_cast[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %reinterpret_cast_2 = memref.reinterpret_cast %alloca_1 to offset: [0], sizes: [%c16], strides: [1] : memref<?xf32> to memref<?xf32>
    %subview_3 = memref.subview %reinterpret_cast_2[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %extracted_slice = tensor.extract_slice %4[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?x?x?x?xf32>
    %extracted_slice_4 = tensor.extract_slice %3[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?x?x?x?xf32>
    %extracted_slice_5 = tensor.extract_slice %2[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : tensor<?x16x16x16xf32> to tensor<?x?x?x?xf32>
    %extracted_slice_6 = tensor.extract_slice %1[0] [%c16] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_7 = tensor.extract_slice %0[0] [%c16] [1] : tensor<?xf32> to tensor<?xf32>
    %8 = linalg.generic {doc = "", indexing_maps = [#map2, #map2, #map2, #map3, #map3, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%subview, %subview_3, %extracted_slice_7, %extracted_slice_4, %extracted_slice_5, %extracted_slice_6 : memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>, tensor<?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?xf32>) outs(%extracted_slice : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_8: f32, %in_9: f32, %in_10: f32, %in_11: f32, %in_12: f32, %out: f32):
      %10 = arith.divf %in_8, %cst_0 : f32
      %11 = arith.subf %in_10, %10 : f32
      %12 = arith.subf %in_11, %in_12 : f32
      %13 = arith.mulf %12, %in_9 : f32
      %14 = arith.mulf %13, %in_9 : f32
      %15 = arith.mulf %14, %in : f32
      %16 = arith.divf %15, %cst_0 : f32
      %17 = arith.subf %11, %16 : f32
      %18 = arith.mulf %in_9, %17 : f32
      linalg.yield %18 : f32
    } -> tensor<?x?x?x?xf32>
    %inserted_slice = tensor.insert_slice %8 into %4[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : tensor<?x?x?x?xf32> into tensor<?x16x16x16xf32>
    %9 = bufferization.to_memref %inserted_slice : memref<?x16x16x16xf32>
    memref.copy %9, %arg4 : memref<?x16x16x16xf32> to memref<?x16x16x16xf32>
    return
  }
}

