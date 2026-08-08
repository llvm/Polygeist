#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2 + d0 * 9 + d1 * 3)>
#map2 = affine_map<(d0, d1, d2) -> (d2 + d1 * 7 + d0 * 42)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map5 = affine_map<(d0) -> (d0 * 2)>
#map6 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = kernel.launch @memset_zero_1D_f32(%1) : (tensor<?xf32>) -> tensor<?xf32>
    %3 = polygeist.submap(%0, %c2, %c3, %c3, %c6, %c6) {map = #map1} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %4 = polygeist.submap(%2, %c2, %c6, %c6) {map = #map2} : (tensor<?xf32>, index, index, index) -> tensor<2x6x6xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"], library_call = ""} ins(%3 : tensor<?x?x?x?x?xf32>) outs(%4 : tensor<2x6x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = linalg.index 2 : index
      %9 = arith.divf %in, %cst_0 : f32
      %10 = arith.addf %out, %9 : f32
      %11 = linalg.index 4 : index
      %12 = affine.apply #map5(%8)
      %13 = arith.cmpi sge, %11, %12 : index
      %14 = affine.apply #map6(%8)
      %15 = arith.cmpi slt, %11, %14 : index
      %16 = arith.andi %13, %15 : i1
      %17 = arith.select %16, %10, %out : f32
      linalg.yield %17 : f32
    } -> tensor<2x6x6xf32>
    %6 = polygeist.submapInverse(%2, %5, %c2, %c6, %c6) {map = #map2} : (tensor<?xf32>, tensor<2x6x6xf32>, index, index, index) -> tensor<?xf32>
    %7 = bufferization.to_memref %6 : memref<?xf32>
    memref.copy %7, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

