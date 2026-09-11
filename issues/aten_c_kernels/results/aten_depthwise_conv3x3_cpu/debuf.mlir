#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (0, d0, d3 + d1 - 1, d4 + d2 - 1)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (-d0 - d1 + 16)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0 + d1 - 1)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 + d3 - 1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (-d2 - d3 + 16)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_depthwise_conv3x3_cpu(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?x3x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?x8x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x8x16x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x3x3xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?x8x16x16xf32>
    %extracted_slice = tensor.extract_slice %2[0] [%c8] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %3[0, 0, 0, 0] [1, %c8, %c16, %c16] [1, 1, 1, 1] : tensor<?x8x16x16xf32> to tensor<?x?x?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_0 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>
    %5 = polygeist.submap(%0, %c8, %c16, %c16, %c3, %c3) {map = #map2} : (tensor<?x8x16x16xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %extracted_slice_1 = tensor.extract_slice %1[0, 0, 0] [%c8, %c3, %c3] [1, 1, 1] : tensor<?x3x3xf32> to tensor<?x?x?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"], library_call = ""} ins(%5, %extracted_slice_1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %8 = linalg.index 1 : index
      %9 = linalg.index 2 : index
      %10 = linalg.index 3 : index
      %11 = linalg.index 4 : index
      %12 = affine.apply #map6(%11, %9, %8, %10)
      %13 = arith.cmpi sge, %12, %c0 : index
      %14 = affine.apply #map7(%11, %9, %8, %10)
      %15 = arith.cmpi sge, %14, %c0 : index
      %16 = arith.andi %13, %15 : i1
      %17 = affine.apply #map8(%11, %9, %8, %10)
      %18 = arith.cmpi sge, %17, %c0 : index
      %19 = arith.andi %16, %18 : i1
      %20 = affine.apply #map9(%11, %9, %8, %10)
      %21 = arith.cmpi sge, %20, %c0 : index
      %22 = arith.andi %19, %21 : i1
      %23 = arith.mulf %in, %in_2 : f32
      %24 = arith.addf %out, %23 : f32
      %25 = arith.select %22, %24, %out : f32
      linalg.yield %25 : f32
    } -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %6 into %3[0, 0, 0, 0] [1, %c8, %c16, %c16] [1, 1, 1, 1] : tensor<?x?x?xf32> into tensor<?x8x16x16xf32>
    %7 = bufferization.to_memref %inserted_slice : memref<?x8x16x16xf32>
    memref.copy %7, %arg3 : memref<?x8x16x16xf32> to memref<?x8x16x16xf32>
    return
  }
}

