#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_gradient_float_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c127 = arith.constant 127 : index
    %c126 = arith.constant 126 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %extracted = tensor.extract %0[%c1] : tensor<?xf32>
    %extracted_0 = tensor.extract %0[%c0] : tensor<?xf32>
    %3 = arith.subf %extracted, %extracted_0 : f32
    %extracted_1 = tensor.extract %1[%c1] : tensor<?xf32>
    %extracted_2 = tensor.extract %1[%c0] : tensor<?xf32>
    %4 = arith.subf %extracted_1, %extracted_2 : f32
    %5 = arith.divf %3, %4 : f32
    %inserted = tensor.insert %5 into %2[%c0] : tensor<?xf32>
    %extracted_slice = tensor.extract_slice %0[2] [%c126] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_3 = tensor.extract_slice %0[0] [%c126] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_4 = tensor.extract_slice %1[2] [%c126] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_5 = tensor.extract_slice %1[0] [%c126] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_6 = tensor.extract_slice %inserted[1] [%c126] [1] : tensor<?xf32> to tensor<?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4, %extracted_slice_5 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_6 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_12: f32, %in_13: f32, %in_14: f32, %out: f32):
      %11 = arith.subf %in, %in_12 : f32
      %12 = arith.subf %in_13, %in_14 : f32
      %13 = arith.divf %11, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %6 into %inserted[1] [%c126] [1] : tensor<?xf32> into tensor<?xf32>
    %extracted_7 = tensor.extract %0[%c127] : tensor<?xf32>
    %extracted_8 = tensor.extract %0[%c126] : tensor<?xf32>
    %7 = arith.subf %extracted_7, %extracted_8 : f32
    %extracted_9 = tensor.extract %1[%c127] : tensor<?xf32>
    %extracted_10 = tensor.extract %1[%c126] : tensor<?xf32>
    %8 = arith.subf %extracted_9, %extracted_10 : f32
    %9 = arith.divf %7, %8 : f32
    %inserted_11 = tensor.insert %9 into %inserted_slice[%c127] : tensor<?xf32>
    %10 = bufferization.to_memref %inserted_11 : memref<?xf32>
    memref.copy %10, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

