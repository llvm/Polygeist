#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3 + d0 * 36 + d1 * 12 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 56 + d0 * 336 + d2 * 8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d5, d6)>
#map5 = affine_map<(d0) -> (d0 * 2)>
#map6 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    %0 = polygeist.submap(%arg0, %c2, %c3, %c3, %c4, %c6, %c6, %c8) {map = #map1} : (memref<?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf32>
    %1 = polygeist.submap(%arg1, %c2, %c6, %c6, %c8) {map = #map2} : (memref<?xf32>, index, index, index, index) -> memref<2x6x6x8xf32>
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction", "reduction", "reduction", "parallel", "parallel", "parallel"]} ins(%0 : memref<?x?x?x?x?x?x?xf32>) outs(%1 : memref<2x6x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = linalg.index 3 : index
      %3 = arith.divf %in, %cst : f32
      %4 = arith.addf %out, %3 : f32
      %5 = linalg.index 6 : index
      %6 = affine.apply #map5(%2)
      %7 = arith.cmpi sge, %5, %6 : index
      %8 = affine.apply #map6(%2)
      %9 = arith.cmpi slt, %5, %8 : index
      %10 = arith.andi %7, %9 : i1
      %11 = arith.select %10, %4, %out : f32
      linalg.yield %11 : f32
    }
    return
  }
}

