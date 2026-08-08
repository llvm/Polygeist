#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_bf16_gemv_trans_cpu(%arg0: memref<?x128xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x128xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%2 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c64, %c128] [1, 1] : tensor<?x128xf32> to tensor<?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_1 = tensor.extract_slice %3[0] [%c128] [1] : tensor<?xf32> to tensor<?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_0 : tensor<?x?xf32>, tensor<?xf32>) outs(%extracted_slice_1 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %4 into %3[0] [%c128] [1] : tensor<?xf32> into tensor<?xf32>
    %5 = bufferization.to_memref %inserted_slice : memref<?xf32>
    memref.copy %5, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

