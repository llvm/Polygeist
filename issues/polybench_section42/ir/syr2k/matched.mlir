#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%0 : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %10 = linalg.index 0 : index
      %11 = arith.mulf %out, %arg3 : f64
      %12 = linalg.index 1 : index
      %13 = affine.apply #map1(%10)
      %14 = arith.cmpi slt, %12, %13 : index
      %15 = arith.select %14, %11, %out : f64
      linalg.yield %15 : f64
    } -> tensor<?x?xf64>
    %6 = arith.subi %4, %c1 : index
    %7 = affine.apply #map1(%6)
    %extracted_slice = tensor.extract_slice %1[0, 0] [%7, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%4, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %2[0, 0] [%7, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %1[0, 0] [%4, %3] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %5[0, 0] [%4, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map2, #map3, #map2, #map3, #map4], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_3 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_4: f64, %in_5: f64, %in_6: f64, %out: f64):
      %10 = linalg.index 0 : index
      %11 = arith.mulf %in, %arg2 : f64
      %12 = arith.mulf %11, %in_4 : f64
      %13 = arith.mulf %in_5, %arg2 : f64
      %14 = arith.mulf %13, %in_6 : f64
      %15 = arith.addf %12, %14 : f64
      %16 = arith.addf %out, %15 : f64
      %17 = linalg.index 2 : index
      %18 = affine.apply #map1(%10)
      %19 = arith.cmpi slt, %17, %18 : index
      %20 = arith.select %19, %16, %out : f64
      linalg.yield %20 : f64
    } -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %8 into %5[0, 0] [%4, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %9 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %9, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

