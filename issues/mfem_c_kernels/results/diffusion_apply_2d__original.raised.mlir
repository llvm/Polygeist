#map = affine_map<(d0, d1) -> (d0, d1, 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1, 0)>
#map3 = affine_map<(d0) -> (d0, 1)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> (d0, 0)>
#map6 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map7 = affine_map<(d0, d1) -> (d1, 1)>
#map8 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0, d1) -> (d1, 0)>
#map10 = affine_map<(d0, d1)[s0] -> (d1 + d0 * 5 + s0 * 75)>
#map11 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 75 + d0 * 5 + 25)>
#map12 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 75 + d0 * 5 + 50)>
#map13 = affine_map<(d0)[s0] -> (d0 * 5 + s0)>
#map14 = affine_map<(d0, d1)[s0] -> (d0 * 5 + s0)>
#map15 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 16 + d0 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %alloca = memref.alloca() : memref<4x2xf64>
    %alloca_0 = memref.alloca() : memref<5x2xf64>
    %alloca_1 = memref.alloca() : memref<5x5x2xf64>
    affine.for %arg7 = 0 to 2 {
      %0 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%0 : memref<?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %1 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map2} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%1 : memref<?x?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      affine.for %arg8 = 0 to 4 {
        %7 = polygeist.submap(%alloca_0, %c5) {map = #map3} : (memref<5x2xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%7 : memref<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %8 = polygeist.submap(%alloca_0, %c5) {map = #map5} : (memref<5x2xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%8 : memref<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 4 {
          %17 = affine.load %arg5[%arg9 + %arg7 * 16 + %arg8 * 4] : memref<?xf64>
          %18 = polygeist.submap(%arg0, %arg9, %c5) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
          %19 = polygeist.submap(%alloca_0, %c5) {map = #map5} : (memref<5x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%18 : memref<?xf64>) outs(%19 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %22 = arith.mulf %17, %in : f64
            %23 = arith.addf %out, %22 : f64
            linalg.yield %23 : f64
          }
          %20 = polygeist.submap(%arg1, %arg9, %c5) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
          %21 = polygeist.submap(%alloca_0, %c5) {map = #map3} : (memref<5x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%20 : memref<?xf64>) outs(%21 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %22 = arith.mulf %17, %in : f64
            %23 = arith.addf %out, %22 : f64
            linalg.yield %23 : f64
          }
        }
        %9 = polygeist.submap(%alloca_0, %c5, %c5) {map = #map7} : (memref<5x2xf64>, index, index) -> memref<?x?xf64>
        %10 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map2} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        %11 = polygeist.submap(%arg0, %arg8, %c5) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
        %12 = polygeist.submap(%11, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%12, %9 : memref<?x?xf64>, memref<?x?xf64>) outs(%10 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
          %17 = arith.mulf %in_2, %in : f64
          %18 = arith.addf %out, %17 : f64
          linalg.yield %18 : f64
        }
        %13 = polygeist.submap(%alloca_0, %c5, %c5) {map = #map9} : (memref<5x2xf64>, index, index) -> memref<?x?xf64>
        %14 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
        %15 = polygeist.submap(%arg1, %arg8, %c5) {map = #map6} : (memref<?xf64>, index, index) -> memref<?xf64>
        %16 = polygeist.submap(%15, %c5, %c5) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%16, %13 : memref<?x?xf64>, memref<?x?xf64>) outs(%14 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
          %17 = arith.mulf %in_2, %in : f64
          %18 = arith.addf %out, %17 : f64
          linalg.yield %18 : f64
        }
      }
      %2 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map10} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %3 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map11} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %4 = polygeist.submap(%arg4, %arg7, %c5, %c5) {map = #map12} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
      %5 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map2} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
      %6 = polygeist.submap(%alloca_1, %c5, %c5) {map = #map} : (memref<5x5x2xf64>, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %3, %4 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%5, %6 : memref<?x?xf64>, memref<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64, %out_4: f64):
        %7 = arith.mulf %in, %out : f64
        %8 = arith.mulf %in_2, %out_4 : f64
        %9 = arith.addf %7, %8 : f64
        %10 = arith.mulf %in_2, %out : f64
        %11 = arith.mulf %in_3, %out_4 : f64
        %12 = arith.addf %10, %11 : f64
        linalg.yield %9, %12 : f64, f64
      }
      affine.for %arg8 = 0 to 5 {
        %7 = polygeist.submap(%alloca, %c4) {map = #map3} : (memref<4x2xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%7 : memref<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        %8 = polygeist.submap(%alloca, %c4) {map = #map5} : (memref<4x2xf64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel"]} outs(%8 : memref<?xf64>) {
        ^bb0(%out: f64):
          linalg.yield %cst : f64
        }
        affine.for %arg9 = 0 to 5 {
          %14 = affine.load %alloca_1[%arg8, %arg9, 0] : memref<5x5x2xf64>
          %15 = affine.load %alloca_1[%arg8, %arg9, 1] : memref<5x5x2xf64>
          %16 = polygeist.submap(%arg3, %arg9, %c4) {map = #map13} : (memref<?xf64>, index, index) -> memref<?xf64>
          %17 = polygeist.submap(%alloca, %c4) {map = #map5} : (memref<4x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%16 : memref<?xf64>) outs(%17 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %20 = arith.mulf %14, %in : f64
            %21 = arith.addf %out, %20 : f64
            linalg.yield %21 : f64
          }
          %18 = polygeist.submap(%arg2, %arg9, %c4) {map = #map13} : (memref<?xf64>, index, index) -> memref<?xf64>
          %19 = polygeist.submap(%alloca, %c4) {map = #map3} : (memref<4x2xf64>, index) -> memref<?xf64>
          linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%18 : memref<?xf64>) outs(%19 : memref<?xf64>) {
          ^bb0(%in: f64, %out: f64):
            %20 = arith.mulf %15, %in : f64
            %21 = arith.addf %out, %20 : f64
            linalg.yield %21 : f64
          }
        }
        %9 = polygeist.submap(%alloca, %c4, %c4) {map = #map9} : (memref<4x2xf64>, index, index) -> memref<?x?xf64>
        %10 = polygeist.submap(%arg2, %arg8, %c4, %c4) {map = #map14} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        %11 = polygeist.submap(%alloca, %c4, %c4) {map = #map7} : (memref<4x2xf64>, index, index) -> memref<?x?xf64>
        %12 = polygeist.submap(%arg3, %arg8, %c4, %c4) {map = #map14} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        %13 = polygeist.submap(%arg6, %arg7, %c4, %c4) {map = #map15} : (memref<?xf64>, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %10, %11, %12 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%13 : memref<?x?xf64>) {
        ^bb0(%in: f64, %in_2: f64, %in_3: f64, %in_4: f64, %out: f64):
          %14 = arith.mulf %in, %in_2 : f64
          %15 = arith.mulf %in_3, %in_4 : f64
          %16 = arith.addf %14, %15 : f64
          %17 = arith.addf %out, %16 : f64
          linalg.yield %17 : f64
        }
      }
    }
    return
  }
}
