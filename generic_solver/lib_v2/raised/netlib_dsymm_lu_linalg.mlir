#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_lu(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = arith.subi %1, %c1 : index
    affine.for %arg10 = 0 to %0 {
      affine.for %arg11 = 0 to %1 {
        %4 = affine.load %arg6[%arg10, %arg11] : memref<?x?xf64>
        %5 = arith.mulf %arg2, %4 : f64
        %alloca = memref.alloca() : memref<f64>
        affine.store %cst, %alloca[] : memref<f64>
        %6 = polygeist.submap(%arg4, %arg11, %2) {map = #map} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%arg9, %arg10, %2) {map = #map} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%6 : memref<?xf64>) outs(%7 : memref<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          %19 = arith.mulf %5, %in : f64
          %20 = arith.addf %out, %19 : f64
          %21 = linalg.index 0 : index
          %22 = arith.cmpi slt, %21, %arg11 : index
          %23 = arith.select %22, %20, %out : f64
          linalg.yield %23 : f64
        }
        %8 = polygeist.submap(%arg6, %arg10, %3) {map = #map} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        %9 = polygeist.submap(%arg4, %arg11, %3) {map = #map} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        %10 = polygeist.submap(%alloca, %3) {map = #map2} : (memref<f64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["reduction"]} ins(%8, %9 : memref<?xf64>, memref<?xf64>) outs(%10 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %19 = arith.mulf %in, %in_0 : f64
          %20 = arith.addf %out, %19 : f64
          %21 = linalg.index 0 : index
          %22 = arith.cmpi slt, %21, %arg11 : index
          %23 = arith.select %22, %20, %out : f64
          linalg.yield %23 : f64
        }
        %11 = affine.load %alloca[] : memref<f64>
        %12 = affine.load %arg9[%arg10, %arg11] : memref<?x?xf64>
        %13 = arith.mulf %arg7, %12 : f64
        %14 = affine.load %arg4[%arg11, %arg11] : memref<?x?xf64>
        %15 = arith.mulf %5, %14 : f64
        %16 = arith.addf %13, %15 : f64
        %17 = arith.mulf %arg2, %11 : f64
        %18 = arith.addf %16, %17 : f64
        affine.store %18, %arg9[%arg10, %arg11] : memref<?x?xf64>
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

