#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_durbin(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = tensor.empty() : tensor<f64>
    %3 = llvm.mlir.undef : f64
    %inserted = tensor.insert %3 into %2[] : tensor<f64>
    %4 = tensor.empty() : tensor<f64>
    %inserted_1 = tensor.insert %3 into %4[] : tensor<f64>
    %5 = tensor.empty() : tensor<f64>
    %inserted_2 = tensor.insert %3 into %5[] : tensor<f64>
    %6 = tensor.empty() : tensor<2000xf64>
    %extracted = tensor.extract %0[%c0] : tensor<?xf64>
    %7 = arith.negf %extracted : f64
    %inserted_3 = tensor.insert %7 into %1[%c0] : tensor<?xf64>
    %inserted_4 = tensor.insert %cst into %inserted_1[] : tensor<f64>
    %extracted_5 = tensor.extract %0[%c0] : tensor<?xf64>
    %8 = arith.negf %extracted_5 : f64
    %inserted_6 = tensor.insert %8 into %inserted_2[] : tensor<f64>
    %9 = arith.index_cast %arg0 : i32 to index
    %10 = arith.subi %9, %c1 : index
    %11 = arith.subi %9, %c1 : index
    %12 = arith.subi %9, %c1 : index
    %13:5 = affine.for %arg3 = 1 to %9 iter_args(%arg4 = %inserted_4, %arg5 = %inserted, %arg6 = %inserted_6, %arg7 = %6, %arg8 = %inserted_3) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<2000xf64>, tensor<?xf64>) {
      %extracted_7 = tensor.extract %arg6[] : tensor<f64>
      %15 = arith.mulf %extracted_7, %extracted_7 : f64
      %16 = arith.subf %cst, %15 : f64
      %extracted_8 = tensor.extract %arg4[] : tensor<f64>
      %17 = arith.mulf %16, %extracted_8 : f64
      %inserted_9 = tensor.insert %17 into %arg4[] : tensor<f64>
      %inserted_10 = tensor.insert %cst_0 into %arg5[] : tensor<f64>
      %18 = polygeist.submap(%0, %arg3, %10) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %19 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"], library_call = ""} ins(%18, %arg8 : tensor<?xf64>, tensor<?xf64>) outs(%inserted_10 : tensor<f64>) {
      ^bb0(%in: f64, %in_19: f64, %out: f64):
        %26 = arith.mulf %in, %in_19 : f64
        %27 = arith.addf %out, %26 : f64
        %28 = linalg.index 0 : index
        %29 = arith.cmpi slt, %28, %arg3 : index
        %30 = arith.select %29, %27, %out : f64
        linalg.yield %30 : f64
      } -> tensor<f64>
      %extracted_11 = tensor.extract %0[%arg3] : tensor<?xf64>
      %extracted_12 = tensor.extract %19[] : tensor<f64>
      %20 = arith.addf %extracted_11, %extracted_12 : f64
      %21 = arith.negf %20 : f64
      %22 = arith.divf %21, %17 : f64
      %inserted_13 = tensor.insert %22 into %arg6[] : tensor<f64>
      %extracted_slice = tensor.extract_slice %arg8[0] [%11] [1] : tensor<?xf64> to tensor<?xf64>
      %23 = polygeist.submap(%arg8, %arg3, %11) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
      %extracted_slice_14 = tensor.extract_slice %arg7[0] [%11] [1] : tensor<2000xf64> to tensor<?xf64>
      %24 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice, %23 : tensor<?xf64>, tensor<?xf64>) outs(%extracted_slice_14 : tensor<?xf64>) {
      ^bb0(%in: f64, %in_19: f64, %out: f64):
        %26 = arith.mulf %22, %in_19 : f64
        %27 = arith.addf %in, %26 : f64
        %28 = linalg.index 0 : index
        %29 = arith.cmpi slt, %28, %arg3 : index
        %30 = arith.select %29, %27, %out : f64
        linalg.yield %30 : f64
      } -> tensor<?xf64>
      %inserted_slice = tensor.insert_slice %24 into %arg7[0] [%11] [1] : tensor<?xf64> into tensor<2000xf64>
      %extracted_slice_15 = tensor.extract_slice %inserted_slice[0] [%12] [1] : tensor<2000xf64> to tensor<?xf64>
      %extracted_slice_16 = tensor.extract_slice %arg8[0] [%12] [1] : tensor<?xf64> to tensor<?xf64>
      %25 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_15 : tensor<?xf64>) outs(%extracted_slice_16 : tensor<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        %26 = linalg.index 0 : index
        %27 = arith.cmpi slt, %26, %arg3 : index
        %28 = arith.select %27, %in, %out : f64
        linalg.yield %28 : f64
      } -> tensor<?xf64>
      %inserted_slice_17 = tensor.insert_slice %25 into %arg8[0] [%12] [1] : tensor<?xf64> into tensor<?xf64>
      %inserted_18 = tensor.insert %22 into %inserted_slice_17[%arg3] : tensor<?xf64>
      affine.yield %inserted_9, %19, %inserted_13, %inserted_slice, %inserted_18 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<2000xf64>, tensor<?xf64>
    }
    %14 = bufferization.to_memref %13#4 : memref<?xf64>
    memref.copy %14, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

