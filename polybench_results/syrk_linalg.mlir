#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x30xf64>, %arg5: memref<?x20xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = affine.apply #map(%2)
    %4 = polygeist.submap(%arg4, %3, %1) {map = #map1} : (memref<?x30xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%4 : memref<?x?xf64>) {
    ^bb0(%out: f64):
      %14 = linalg.index 0 : index
      %15 = arith.mulf %out, %arg3 : f64
      %16 = linalg.index 1 : index
      %17 = affine.apply #map(%14)
      %18 = arith.cmpi slt, %16, %17 : index
      %19 = arith.select %18, %15, %out : f64
      linalg.yield %19 : f64
    }
    %5 = arith.subi %1, %c1 : index
    %6 = affine.apply #map(%5)
    %7 = polygeist.submap(%arg5, %6, %0, %1) {map = #map3} : (memref<?x20xf64>, index, index, index) -> memref<?x?x?xf64>
    %8 = arith.subi %1, %c1 : index
    %9 = affine.apply #map(%8)
    %10 = polygeist.submap(%arg5, %9, %0, %1) {map = #map4} : (memref<?x20xf64>, index, index, index) -> memref<?x?x?xf64>
    %11 = arith.subi %1, %c1 : index
    %12 = affine.apply #map(%11)
    %13 = polygeist.submap(%arg4, %12, %0, %1) {map = #map5} : (memref<?x30xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "reduction", "parallel"]} ins(%7, %10 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%13 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %14 = linalg.index 0 : index
      %15 = arith.mulf %arg2, %in : f64
      %16 = arith.mulf %15, %in_0 : f64
      %17 = arith.addf %out, %16 : f64
      %18 = linalg.index 2 : index
      %19 = affine.apply #map(%14)
      %20 = arith.cmpi slt, %18, %19 : index
      %21 = arith.select %20, %17, %out : f64
      linalg.yield %21 : f64
    }
    return
  }
}

