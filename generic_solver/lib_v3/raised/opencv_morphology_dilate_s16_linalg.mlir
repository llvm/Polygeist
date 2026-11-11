#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d2 + d0 + 1 - 1, d3 + d1 + 1 - 1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_morphology_dilate_s16(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi16>, %arg3: memref<?x?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c-32768_i16 = arith.constant -32768 : i16
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = affine.apply #map()[%1]
    %3 = affine.apply #map()[%1]
    %4 = arith.subi %3, %c1 : index
    %5 = arith.subi %2, %c1 : index
    %6 = affine.apply #map()[%0]
    %7 = arith.subi %6, %c1 : index
    %8 = polygeist.submap(%arg3, %7, %4) {map = #map1} : (memref<?x?xi16>, index, index) -> memref<?x?xi16>
    linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%8 : memref<?x?xi16>) {
    ^bb0(%out: i16):
      linalg.yield %c-32768_i16 : i16
    }
    %9 = affine.apply #map()[%0]
    %10 = arith.subi %9, %c1 : index
    %11 = polygeist.submap(%arg2, %10, %5, %c3, %c3) {map = #map3} : (memref<?x?xi16>, index, index, index, index) -> memref<?x?x?x?xi16>
    %12 = polygeist.submap(%arg3, %10, %5, %c3, %c3) {map = #map4} : (memref<?x?xi16>, index, index, index, index) -> memref<?x?x?x?xi16>
    linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%11 : memref<?x?x?x?xi16>) outs(%12 : memref<?x?x?x?xi16>) {
    ^bb0(%in: i16, %out: i16):
      %13 = arith.extsi %out : i16 to i32
      %14 = arith.extsi %in : i16 to i32
      %15 = arith.cmpi sgt, %13, %14 : i32
      %16 = arith.select %15, %13, %14 : i32
      %17 = arith.trunci %16 : i32 to i16
      linalg.yield %17 : i16
    }
    return
  }
}

