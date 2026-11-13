#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map3 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_ludcmp(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = bufferization.to_tensor %arg4 : memref<?xf64>
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = tensor.empty() : tensor<f64>
    %6 = llvm.mlir.undef : f64
    %inserted = tensor.insert %6 into %5[] : tensor<f64>
    %7 = arith.subi %4, %c1 : index
    %8:2 = affine.for %arg5 = 0 to %4 iter_args(%arg6 = %inserted, %arg7 = %0) -> (tensor<f64>, tensor<?x?xf64>) {
      %17 = arith.subi %arg5, %c1 : index
      %18:2 = affine.for %arg8 = 0 to #map(%arg5) iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (tensor<f64>, tensor<?x?xf64>) {
        %extracted = tensor.extract %arg10[%arg5, %arg8] : tensor<?x?xf64>
        %inserted_0 = tensor.insert %extracted into %arg9[] : tensor<f64>
        %extracted_slice = tensor.extract_slice %arg10[%arg5, 0] [1, %17] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_1 = tensor.extract_slice %arg10[0, %arg8] [%17, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %20 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_1 : tensor<?xf64>, tensor<?xf64>) outs(%inserted_0 : tensor<f64>) {
        ^bb0(%in: f64, %in_5: f64, %out: f64):
          %22 = arith.mulf %in, %in_5 : f64
          %23 = arith.subf %out, %22 : f64
          %24 = linalg.index 0 : index
          %25 = arith.cmpi slt, %24, %arg8 : index
          %26 = arith.select %25, %23, %out : f64
          linalg.yield %26 : f64
        } -> tensor<f64>
        %extracted_2 = tensor.extract %20[] : tensor<f64>
        %extracted_3 = tensor.extract %arg10[%arg8, %arg8] : tensor<?x?xf64>
        %21 = arith.divf %extracted_2, %extracted_3 : f64
        %inserted_4 = tensor.insert %21 into %arg10[%arg5, %arg8] : tensor<?x?xf64>
        affine.yield %20, %inserted_4 : tensor<f64>, tensor<?x?xf64>
      }
      %19:2 = affine.for %arg8 = #map(%arg5) to %4 iter_args(%arg9 = %18#0, %arg10 = %18#1) -> (tensor<f64>, tensor<?x?xf64>) {
        %extracted = tensor.extract %arg10[%arg5, %arg8] : tensor<?x?xf64>
        %inserted_0 = tensor.insert %extracted into %arg9[] : tensor<f64>
        %extracted_slice = tensor.extract_slice %arg10[%arg5, 0] [1, %7] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_1 = tensor.extract_slice %arg10[0, %arg8] [%7, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %20 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_1 : tensor<?xf64>, tensor<?xf64>) outs(%inserted_0 : tensor<f64>) {
        ^bb0(%in: f64, %in_4: f64, %out: f64):
          %21 = arith.mulf %in, %in_4 : f64
          %22 = arith.subf %out, %21 : f64
          %23 = linalg.index 0 : index
          %24 = arith.cmpi slt, %23, %arg5 : index
          %25 = arith.select %24, %22, %out : f64
          linalg.yield %25 : f64
        } -> tensor<f64>
        %extracted_2 = tensor.extract %20[] : tensor<f64>
        %inserted_3 = tensor.insert %extracted_2 into %arg10[%arg5, %arg8] : tensor<?x?xf64>
        affine.yield %20, %inserted_3 : tensor<f64>, tensor<?x?xf64>
      }
      affine.yield %19#0, %19#1 : tensor<f64>, tensor<?x?xf64>
    }
    %9 = bufferization.to_memref %8#1 : memref<?x?xf64>
    memref.copy %9, %arg1 : memref<?x?xf64> to memref<?x?xf64>
    %10 = arith.subi %4, %c1 : index
    %11 = tensor.empty(%4) : tensor<?xf64>
    %12:2 = affine.for %arg5 = 0 to %4 iter_args(%arg6 = %11, %arg7 = %3) -> (tensor<?xf64>, tensor<?xf64>) {
      %extracted = tensor.extract %1[%arg5] : tensor<?xf64>
      %inserted_0 = tensor.insert %extracted into %arg6[%arg5] : tensor<?xf64>
      %extracted_slice = tensor.extract_slice %8#1[%arg5, 0] [1, %10] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
      %extracted_slice_1 = tensor.extract_slice %arg7[0] [%10] [1] : tensor<?xf64> to tensor<?xf64>
      %extracted_slice_2 = tensor.extract_slice %inserted_0[%arg5] [1] [1] : tensor<?xf64> to tensor<f64>
      %17 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_1 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_2 : tensor<f64>) {
      ^bb0(%in: f64, %in_5: f64, %out: f64):
        %18 = arith.mulf %in, %in_5 : f64
        %19 = arith.subf %out, %18 : f64
        %20 = linalg.index 0 : index
        %21 = arith.cmpi slt, %20, %arg5 : index
        %22 = arith.select %21, %19, %out : f64
        linalg.yield %22 : f64
      } -> tensor<f64>
      %inserted_slice = tensor.insert_slice %17 into %inserted_0[%arg5] [1] [1] : tensor<f64> into tensor<?xf64>
      %extracted_3 = tensor.extract %inserted_slice[%arg5] : tensor<?xf64>
      %inserted_4 = tensor.insert %extracted_3 into %arg7[%arg5] : tensor<?xf64>
      affine.yield %inserted_slice, %inserted_4 : tensor<?xf64>, tensor<?xf64>
    }
    %13 = bufferization.to_memref %12#1 : memref<?xf64>
    memref.copy %13, %arg4 : memref<?xf64> to memref<?xf64>
    %14 = tensor.empty(%4) : tensor<?xf64>
    %15:2 = affine.for %arg5 = 0 to %4 iter_args(%arg6 = %14, %arg7 = %2) -> (tensor<?xf64>, tensor<?xf64>) {
      %17 = affine.apply #map2(%arg5)[%4]
      %extracted = tensor.extract %12#1[%17] : tensor<?xf64>
      %inserted_0 = tensor.insert %extracted into %arg6[%arg5] : tensor<?xf64>
      %18 = polygeist.submap(%8#1, %arg5, %4, %4) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
      %extracted_slice = tensor.extract_slice %arg7[0] [%4] [1] : tensor<?xf64> to tensor<?xf64>
      %extracted_slice_1 = tensor.extract_slice %inserted_0[%arg5] [1] [1] : tensor<?xf64> to tensor<f64>
      %19 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%18, %extracted_slice : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_1 : tensor<f64>) {
      ^bb0(%in: f64, %in_5: f64, %out: f64):
        %24 = arith.mulf %in, %in_5 : f64
        %25 = arith.subf %out, %24 : f64
        %26 = linalg.index 0 : index
        %27 = affine.apply #map4(%arg5)[%4]
        %28 = arith.cmpi sge, %26, %27 : index
        %29 = arith.select %28, %25, %out : f64
        linalg.yield %29 : f64
      } -> tensor<f64>
      %inserted_slice = tensor.insert_slice %19 into %inserted_0[%arg5] [1] [1] : tensor<f64> into tensor<?xf64>
      %extracted_2 = tensor.extract %inserted_slice[%arg5] : tensor<?xf64>
      %20 = affine.apply #map2(%arg5)[%4]
      %21 = affine.apply #map2(%arg5)[%4]
      %extracted_3 = tensor.extract %8#1[%20, %21] : tensor<?x?xf64>
      %22 = arith.divf %extracted_2, %extracted_3 : f64
      %23 = affine.apply #map2(%arg5)[%4]
      %inserted_4 = tensor.insert %22 into %arg7[%23] : tensor<?xf64>
      affine.yield %inserted_slice, %inserted_4 : tensor<?xf64>, tensor<?xf64>
    }
    %16 = bufferization.to_memref %15#1 : memref<?xf64>
    memref.copy %16, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}

