#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4 + d1 - 1, d5 + d2 - 1, d6 + d3 - 1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d5, d6, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (-d0 - d1 + 10)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d1 - 1)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (-d2 - d3 + 9)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2 + d3 - 1)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4 + d5 - 1)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (-d4 - d5 + 8)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unfold3d_zero_copy_cpu(%arg0: memref<?x8x9x10xf32>, %arg1: memref<?x3x3x3x8x9x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c3 = arith.constant 3 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x8x9x10xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x3x3x3x8x9x10xf32>
    %2 = polygeist.submap(%0, %c2, %c8, %c9, %c10, %c3, %c3, %c3) {map = #map} : (tensor<?x8x9x10xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0, 0, 0, 0, 0, 0] [%c2, %c3, %c3, %c3, %c8, %c9, %c10] [1, 1, 1, 1, 1, 1, 1] : tensor<?x3x3x3x8x9x10xf32> to tensor<?x?x?x?x?x?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%2 : tensor<?x?x?x?x?x?x?xf32>) outs(%extracted_slice : tensor<?x?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = linalg.index 1 : index
      %6 = linalg.index 2 : index
      %7 = linalg.index 3 : index
      %8 = linalg.index 4 : index
      %9 = linalg.index 5 : index
      %10 = linalg.index 6 : index
      %11 = affine.apply #map3(%10, %7, %9, %6, %5, %8)
      %12 = arith.cmpi sge, %11, %c0 : index
      %13 = affine.apply #map4(%10, %7, %9, %6, %5, %8)
      %14 = arith.cmpi sge, %13, %c0 : index
      %15 = arith.andi %12, %14 : i1
      %16 = affine.apply #map5(%10, %7, %9, %6, %5, %8)
      %17 = arith.cmpi sge, %16, %c0 : index
      %18 = arith.andi %15, %17 : i1
      %19 = affine.apply #map6(%10, %7, %9, %6, %5, %8)
      %20 = arith.cmpi sge, %19, %c0 : index
      %21 = arith.andi %18, %20 : i1
      %22 = affine.apply #map7(%10, %7, %9, %6, %5, %8)
      %23 = arith.cmpi sge, %22, %c0 : index
      %24 = arith.andi %21, %23 : i1
      %25 = affine.apply #map8(%10, %7, %9, %6, %5, %8)
      %26 = arith.cmpi sge, %25, %c0 : index
      %27 = arith.andi %24, %26 : i1
      %28 = arith.select %27, %in, %cst : f32
      linalg.yield %28 : f32
    } -> tensor<?x?x?x?x?x?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0, 0, 0, 0, 0, 0] [%c2, %c3, %c3, %c3, %c8, %c9, %c10] [1, 1, 1, 1, 1, 1, 1] : tensor<?x?x?x?x?x?x?xf32> into tensor<?x3x3x3x8x9x10xf32>
    %4 = bufferization.to_memref %inserted_slice : memref<?x3x3x3x8x9x10xf32>
    memref.copy %4, %arg1 : memref<?x3x3x3x8x9x10xf32> to memref<?x3x3x3x8x9x10xf32>
    return
  }
}

