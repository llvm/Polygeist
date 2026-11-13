#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2)[s0] -> (s0, d1)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1] -> (d2 * 125 + d1 * 375 + s0 * 1125 + s1)>
#map4 = affine_map<(d0, d1, d2)[s0, s1] -> (d2 * 375 + s0 * 1125 + s1 + d1 * 125)>
#map5 = affine_map<(d0, d1, d2) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0)[s0] -> (s0, d0)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<(d0, d1)[s0, s1] -> (d1 * 375 + s0 * 1125 + s1 + d0 * 125)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<3x3xf64>
    %alloca_1 = memref.alloca() : memref<3x3xf64>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 125 {
        %0 = affine.load %arg2[%arg6 + %arg5 * 1125] : memref<?xf64>
        %1 = affine.load %arg2[%arg6 + %arg5 * 1125 + 125] : memref<?xf64>
        %2 = affine.load %arg2[%arg6 + %arg5 * 1125 + 250] : memref<?xf64>
        %3 = affine.load %arg2[%arg6 + %arg5 * 1125 + 375] : memref<?xf64>
        %4 = affine.load %arg2[%arg6 + %arg5 * 1125 + 500] : memref<?xf64>
        %5 = affine.load %arg2[%arg6 + %arg5 * 1125 + 625] : memref<?xf64>
        %6 = affine.load %arg2[%arg6 + %arg5 * 1125 + 750] : memref<?xf64>
        %7 = affine.load %arg2[%arg6 + %arg5 * 1125 + 875] : memref<?xf64>
        %8 = affine.load %arg2[%arg6 + %arg5 * 1125 + 1000] : memref<?xf64>
        %9 = arith.mulf %4, %8 : f64
        %10 = arith.mulf %5, %7 : f64
        %11 = arith.subf %9, %10 : f64
        %12 = arith.mulf %0, %11 : f64
        %13 = arith.mulf %3, %8 : f64
        %14 = arith.mulf %5, %6 : f64
        %15 = arith.subf %13, %14 : f64
        %16 = arith.mulf %1, %15 : f64
        %17 = arith.subf %12, %16 : f64
        %18 = arith.mulf %3, %7 : f64
        %19 = arith.mulf %4, %6 : f64
        %20 = arith.subf %18, %19 : f64
        %21 = arith.mulf %2, %20 : f64
        %22 = arith.addf %17, %21 : f64
        %23 = arith.divf %11, %22 : f64
        affine.store %23, %alloca_1[0, 0] : memref<3x3xf64>
        %24 = arith.mulf %2, %7 : f64
        %25 = arith.mulf %1, %8 : f64
        %26 = arith.subf %24, %25 : f64
        %27 = arith.divf %26, %22 : f64
        affine.store %27, %alloca_1[0, 1] : memref<3x3xf64>
        %28 = arith.mulf %1, %5 : f64
        %29 = arith.mulf %2, %4 : f64
        %30 = arith.subf %28, %29 : f64
        %31 = arith.divf %30, %22 : f64
        affine.store %31, %alloca_1[0, 2] : memref<3x3xf64>
        %32 = arith.subf %14, %13 : f64
        %33 = arith.divf %32, %22 : f64
        affine.store %33, %alloca_1[1, 0] : memref<3x3xf64>
        %34 = arith.mulf %0, %8 : f64
        %35 = arith.mulf %2, %6 : f64
        %36 = arith.subf %34, %35 : f64
        %37 = arith.divf %36, %22 : f64
        affine.store %37, %alloca_1[1, 1] : memref<3x3xf64>
        %38 = arith.mulf %2, %3 : f64
        %39 = arith.mulf %0, %5 : f64
        %40 = arith.subf %38, %39 : f64
        %41 = arith.divf %40, %22 : f64
        affine.store %41, %alloca_1[1, 2] : memref<3x3xf64>
        %42 = arith.divf %20, %22 : f64
        affine.store %42, %alloca_1[2, 0] : memref<3x3xf64>
        %43 = arith.mulf %1, %6 : f64
        %44 = arith.mulf %0, %7 : f64
        %45 = arith.subf %43, %44 : f64
        %46 = arith.divf %45, %22 : f64
        affine.store %46, %alloca_1[2, 1] : memref<3x3xf64>
        %47 = arith.mulf %0, %4 : f64
        %48 = arith.mulf %1, %3 : f64
        %49 = arith.subf %47, %48 : f64
        %50 = arith.divf %49, %22 : f64
        affine.store %50, %alloca_1[2, 2] : memref<3x3xf64>
        %51 = affine.load %arg4[%arg6 + %arg5 * 1125] : memref<?xf64>
        %52 = affine.load %arg4[%arg6 + %arg5 * 1125 + 500] : memref<?xf64>
        %53 = arith.addf %51, %52 : f64
        %54 = affine.load %arg4[%arg6 + %arg5 * 1125 + 1000] : memref<?xf64>
        %55 = arith.addf %53, %54 : f64
        %56 = affine.load %arg3[%arg6] : memref<?xf64>
        %57 = arith.mulf %56, %22 : f64
        %58 = affine.load %arg0[%arg6 + %arg5 * 125] : memref<?xf64>
        %59 = affine.load %arg1[%arg6 + %arg5 * 125] : memref<?xf64>
        %60 = arith.mulf %59, %cst : f64
        affine.for %arg7 = 0 to 3 {
          %alloca_2 = memref.alloca(%c3) : memref<?xf64>
          %63 = polygeist.submap(%alloca_2, %c3) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%63 : memref<?xf64>) {
          ^bb0(%out: f64):
            linalg.yield %cst_0 : f64
          }
          %64 = polygeist.submap(%alloca_1, %arg7, %c3, %c3, %c3) {map = #map1} : (memref<3x3xf64>, index, index, index, index) -> memref<?x?x?xf64>
          %65 = polygeist.submap(%alloca_1, %arg7, %c3, %c3, %c3) {map = #map2} : (memref<3x3xf64>, index, index, index, index) -> memref<?x?x?xf64>
          %66 = polygeist.submap(%arg4, %arg5, %arg6, %c3, %c3, %c3) {map = #map3} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?xf64>
          %67 = polygeist.submap(%arg4, %arg5, %arg6, %c3, %c3, %c3) {map = #map4} : (memref<?xf64>, index, index, index, index, index) -> memref<?x?x?xf64>
          %68 = polygeist.submap(%alloca_2, %c3, %c3, %c3) {map = #map5} : (memref<?xf64>, index, index, index) -> memref<?x?x?xf64>
          linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction", "reduction"]} ins(%64, %65, %66, %67 : memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%68 : memref<?x?x?xf64>) {
          ^bb0(%in: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
            %72 = linalg.index 0 : index
            %73 = arith.index_cast %72 : index to i32
            %74 = linalg.index 1 : index
            %75 = arith.index_cast %74 : index to i32
            %76 = arith.cmpi eq, %75, %73 : i32
            %77 = arith.extui %76 : i1 to i32
            %78 = arith.sitofp %77 : i32 to f64
            %79 = linalg.index 2 : index
            %80 = arith.index_cast %79 : index to i32
            %81 = arith.mulf %78, %in_3 : f64
            %82 = arith.cmpi eq, %80, %73 : i32
            %83 = arith.extui %82 : i1 to i32
            %84 = arith.sitofp %83 : i32 to f64
            %85 = arith.mulf %84, %in : f64
            %86 = arith.addf %81, %85 : f64
            %87 = arith.addf %in_4, %in_5 : f64
            %88 = arith.mulf %86, %87 : f64
            %89 = arith.addf %out, %88 : f64
            linalg.yield %89 : f64
          }
          %69 = polygeist.submap(%alloca_2, %c3) {map = #map} : (memref<?xf64>, index) -> memref<?xf64>
          %70 = polygeist.submap(%alloca_1, %arg7, %c3) {map = #map7} : (memref<3x3xf64>, index, index) -> memref<?xf64>
          %71 = polygeist.submap(%alloca, %arg7, %c3) {map = #map7} : (memref<3x3xf64>, index, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%69, %70 : memref<?xf64>, memref<?xf64>) outs(%71 : memref<?xf64>) {
          ^bb0(%in: f64, %in_3: f64, %out: f64):
            %72 = arith.mulf %58, %in_3 : f64
            %73 = arith.mulf %72, %55 : f64
            %74 = arith.mulf %60, %in : f64
            %75 = arith.addf %73, %74 : f64
            %76 = arith.mulf %57, %75 : f64
            linalg.yield %76 : f64
          }
        } {polygeist.was_parallel}
        %61 = polygeist.submap(%alloca, %c3, %c3) {map = #map8} : (memref<3x3xf64>, index, index) -> memref<?x?xf64>
        %62 = polygeist.submap(%arg4, %arg5, %arg6, %c3, %c3) {map = #map9} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel"]} ins(%61 : memref<?x?xf64>) outs(%62 : memref<?x?xf64>) {
        ^bb0(%in: f64, %out: f64):
          linalg.yield %in : f64
        }
      }
    }
    return
  }
}
