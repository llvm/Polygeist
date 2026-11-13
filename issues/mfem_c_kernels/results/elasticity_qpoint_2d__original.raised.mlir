#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2)[s0] -> (s0, d1)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1] -> (d2 * 25 + d1 * 50 + s0 * 100 + s1)>
#map4 = affine_map<(d0, d1, d2)[s0, s1] -> (d2 * 50 + s0 * 100 + s1 + d1 * 25)>
#map5 = affine_map<(d0, d1, d2) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0)[s0] -> (s0, d0)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<(d0, d1)[s0, s1] -> (d1 * 50 + s0 * 100 + s1 + d0 * 25)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<2x2xf64>
    %alloca_1 = memref.alloca() : memref<2x2xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 25 {
        %0 = affine.load %arg2[%arg6 + %arg5 * 100] : memref<?xf64>
        %1 = affine.load %arg2[%arg6 + %arg5 * 100 + 25] : memref<?xf64>
        %2 = affine.load %arg2[%arg6 + %arg5 * 100 + 50] : memref<?xf64>
        %3 = affine.load %arg2[%arg6 + %arg5 * 100 + 75] : memref<?xf64>
        %4 = arith.mulf %0, %3 : f64
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.subf %4, %5 : f64
        %7 = arith.divf %3, %6 : f64
        affine.store %7, %alloca_1[0, 0] : memref<2x2xf64>
        %8 = arith.negf %1 : f64
        %9 = arith.divf %8, %6 : f64
        affine.store %9, %alloca_1[0, 1] : memref<2x2xf64>
        %10 = arith.negf %2 : f64
        %11 = arith.divf %10, %6 : f64
        affine.store %11, %alloca_1[1, 0] : memref<2x2xf64>
        %12 = arith.divf %0, %6 : f64
        affine.store %12, %alloca_1[1, 1] : memref<2x2xf64>
        %13 = affine.load %arg4[%arg6 + %arg5 * 100] : memref<?xf64>
        %14 = affine.load %arg4[%arg6 + %arg5 * 100 + 75] : memref<?xf64>
        %15 = arith.addf %13, %14 : f64
        %16 = affine.load %arg3[%arg6] : memref<?xf64>
        %17 = arith.mulf %16, %6 : f64
        %18 = affine.load %arg0[%arg6 + %arg5 * 25] : memref<?xf64>
        %19 = affine.load %arg1[%arg6 + %arg5 * 25] : memref<?xf64>
        %20 = arith.mulf %19, %cst : f64
        affine.for %arg7 = 0 to 2 {
          %alloca_2 = memref.alloca(%c2) : memref<?xf64>
          %23 = polygeist.submap(%alloca_2, %c2) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%23 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst_0 : f64
          }
          %24 = polygeist.submap(%alloca_1, %arg7, %c2, %c2, %c2) {map = #map1} : (memref<2x2xf64>, index, index, index, index) -> memref<?x?x?xf64>
          %25 = polygeist.submap(%alloca_1, %arg7, %c2, %c2, %c2) {map = #map2} : (memref<2x2xf64>, index, index, index, index) -> memref<?x?x?xf64>
          %26 = polygeist.submap(%arg4, %arg5, %arg6, %c2, %c2, %c2) {map = #map3} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?xf64>
          %27 = polygeist.submap(%arg4, %arg5, %arg6, %c2, %c2, %c2) {map = #map4} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?xf64>
          %28 = polygeist.submap(%alloca_2, %c2, %c2, %c2) {map = #map5} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction", "reduction"]} ins(%24, %25, %26, %27 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%28 : memref<?x?x?xf64>) {
          ^bb0(%in: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
            %32 = linalg.index 0 : index
            %33 = arith.index_cast %32 : index to i32
            %34 = linalg.index 1 : index
            %35 = arith.index_cast %34 : index to i32
            %36 = arith.cmpi eq, %35, %33 : i32
            %37 = arith.extui %36 : i1 to i32
            %38 = arith.sitofp %37 : i32 to f64
            %39 = linalg.index 2 : index
            %40 = arith.index_cast %39 : index to i32
            %41 = arith.mulf %38, %in_3 : f64
            %42 = arith.cmpi eq, %40, %33 : i32
            %43 = arith.extui %42 : i1 to i32
            %44 = arith.sitofp %43 : i32 to f64
            %45 = arith.mulf %44, %in : f64
            %46 = arith.addf %41, %45 : f64
            %47 = arith.addf %in_4, %in_5 : f64
            %48 = arith.mulf %46, %47 : f64
            %49 = arith.addf %out, %48 : f64
            linalg.yield %49 : f64
          }
          %29 = polygeist.submap(%alloca_2, %c2) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %30 = polygeist.submap(%alloca_1, %arg7, %c2) {map = #map7} : (memref<2x2xf64>, index, index) -> memref<?xf64>
          %31 = polygeist.submap(%alloca, %arg7, %c2) {map = #map7} : (memref<2x2xf64>, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%29, %30 : memref<?xf64>, memref<?xf64>) outs(%31 : memref<?xf64>) {
          ^bb0(%in: f64, %in_3: f64, %out: f64):
            %32 = arith.mulf %18, %in_3 : f64
            %33 = arith.mulf %32, %15 : f64
            %34 = arith.mulf %20, %in : f64
            %35 = arith.addf %33, %34 : f64
            %36 = arith.mulf %17, %35 : f64
            linalg.yield %36 : f64
          }
        } {polygeist.was_parallel}
        %21 = polygeist.submap(%alloca, %c2, %c2) {map = #map8} : (memref<2x2xf64>, index, index) -> memref<?x?xf64>
        %22 = polygeist.submap(%arg4, %arg5, %arg6, %c2, %c2) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%21 : memref<?x?xf64>) outs(%22 : memref<?x?xf64>) {
        ^bb0(%in: f64, %out: f64):
          linalg.yield %in : f64
        }
      }
    }
    return
  }
}
