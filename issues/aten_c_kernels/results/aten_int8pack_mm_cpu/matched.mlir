#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_int8pack_mm_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xi8>, %arg2: memref<?xf32>, %arg3: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c48 = arith.constant 48 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x64xi8>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?x48xf32>
    %extracted_slice = tensor.extract_slice %3[0, 0] [%c32, %c48] [1, 1] : tensor<?x48xf32> to tensor<?x?xf32>
    %4 = kernel.launch @memset_zero_2D_f32(%extracted_slice) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, 0] [%c32, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_1 = tensor.extract_slice %1[0, 0] [%c48, %c64] [1, 1] : tensor<?x64xi8> to tensor<?x?xi8>
    %extracted_slice_2 = tensor.extract_slice %2[0] [%c48] [1] : tensor<?xf32> to tensor<?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map1, #map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%extracted_slice_0, %extracted_slice_1, %extracted_slice_2 : tensor<?x?xf32>, tensor<?x?xi8>, tensor<?xf32>) outs(%4 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_3: i8, %in_4: f32, %out: f32):
      %7 = arith.sitofp %in_3 : i8 to f32
      %8 = arith.mulf %in, %7 : f32
      %9 = arith.mulf %8, %in_4 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %5 into %3[0, 0] [%c32, %c48] [1, 1] : tensor<?x?xf32> into tensor<?x48xf32>
    %6 = bufferization.to_memref %inserted_slice : memref<?x48xf32>
    memref.copy %6, %arg3 : memref<?x48xf32> to memref<?x48xf32>
    return
  }
}

