#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_softmax_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.load %arg1[0] : memref<?xf32>
    %1 = arith.index_cast %arg0 : i32 to index
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %2 = arith.subi %1, %c1 : index
    %3 = polygeist.submap(%arg1, %2) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %4 = polygeist.submap(%alloca, %2) {map = #map1} : (memref<f32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["reduction"]} ins(%3 : memref<?xf32>) outs(%4 : memref<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.cmpf olt, %out, %in : f32
      %12 = arith.select %11, %in, %out : f32
      linalg.yield %12 : f32
    }
    %5 = affine.load %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    affine.store %cst, %alloca_0[] : memref<f32>
    %6 = polygeist.submap(%arg1, %1) {map = #map2} : (memref<?xf32>, index) -> memref<?xf32>
    %7 = polygeist.submap(%arg2, %1) {map = #map2} : (memref<?xf32>, index) -> memref<?xf32>
    %8 = polygeist.submap(%alloca_0, %1) {map = #map1} : (memref<f32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["reduction"]} ins(%6 : memref<?xf32>) outs(%7, %8 : memref<?xf32>, memref<?xf32>) {
    ^bb0(%in: f32, %out: f32, %out_1: f32):
      %11 = arith.subf %in, %5 : f32
      %12 = math.exp %11 : f32
      %13 = arith.addf %out_1, %12 : f32
      linalg.yield %12, %13 : f32, f32
    }
    %9 = affine.load %alloca_0[] : memref<f32>
    %10 = polygeist.submap(%arg2, %1) {map = #map2} : (memref<?xf32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%10 : memref<?xf32>) {
    ^bb0(%out: f32):
      %11 = arith.divf %out, %9 : f32
      linalg.yield %11 : f32
    }
    return
  }
}

