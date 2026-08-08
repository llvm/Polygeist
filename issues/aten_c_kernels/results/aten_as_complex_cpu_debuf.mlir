#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_as_complex_cpu(%arg0: memref<?x2xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c512 = arith.constant 512 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x2xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c512, 1] [1, 1] : tensor<?x2xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0] [%c512] [1] : tensor<?xf32> to tensor<?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_0 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0] [%c512] [1] : tensor<?xf32> into tensor<?xf32>
    %4 = bufferization.to_memref %inserted_slice : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    %extracted_slice_1 = tensor.extract_slice %0[0, 1] [%c512, 1] [1, 1] : tensor<?x2xf32> to tensor<?xf32>
    %extracted_slice_2 = tensor.extract_slice %2[0] [%c512] [1] : tensor<?xf32> to tensor<?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_1 : tensor<?xf32>) outs(%extracted_slice_2 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?xf32>
    %inserted_slice_3 = tensor.insert_slice %5 into %2[0] [%c512] [1] : tensor<?xf32> into tensor<?xf32>
    %6 = bufferization.to_memref %inserted_slice_3 : memref<?xf32>
    memref.copy %6, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

