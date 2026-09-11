#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
#map2 = affine_map<(d0) -> (0, d0 + 1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> (d0 + 1, 0)>
#map5 = affine_map<(d0)[s0] -> (s0 - 1, d0 + 1)>
#map6 = affine_map<(d0, d1)[s0] -> (d1 + 1, -(d0 + 1) + s0 - 1)>
#map7 = affine_map<(d0, d1)[s0] -> (-(d0 + 1) + s0, d1 + 1)>
#map8 = affine_map<(d0, d1)[s0] -> (-(d0 + 1) + s0 - 1, d1 + 1)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0)[s0] -> (d0 + 1, s0 - 1)>
#map11 = affine_map<(d0, d1)[s0] -> (d1 + 1, -(d0 + 1) + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<?x20xf64>, %arg3: memref<?x20xf64>, %arg4: memref<?x20xf64>, %arg5: memref<?x20xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.sitofp %arg1 : i32 to f64
    %2 = arith.divf %cst, %1 : f64
    %3 = arith.sitofp %arg0 : i32 to f64
    %4 = arith.divf %cst, %3 : f64
    %5 = arith.mulf %4, %cst_0 : f64
    %6 = arith.mulf %2, %2 : f64
    %7 = arith.divf %5, %6 : f64
    %8 = arith.divf %4, %6 : f64
    %9 = arith.negf %7 : f64
    %10 = arith.divf %9, %cst_0 : f64
    %11 = arith.addf %7, %cst : f64
    %12 = arith.negf %8 : f64
    %13 = arith.divf %12, %cst_0 : f64
    %14 = arith.addf %8, %cst : f64
    %15 = arith.index_cast %arg0 : i32 to index
    %16 = arith.negf %10 : f64
    %17 = arith.negf %13 : f64
    %18 = arith.mulf %13, %cst_0 : f64
    %19 = arith.addf %18, %cst : f64
    %20 = arith.mulf %10, %cst_0 : f64
    %21 = arith.addf %20, %cst : f64
    affine.for %arg6 = 1 to #map()[%15] {
      %22 = affine.apply #map1()[%0]
      %23 = arith.subi %22, %c1 : index
      %24 = polygeist.submap(%arg3, %23) {map = #map2} : (memref<?x20xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%24 : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %25 = affine.apply #map1()[%0]
      %26 = arith.subi %25, %c1 : index
      %27 = polygeist.submap(%arg4, %26) {map = #map4} : (memref<?x20xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%27 : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      }
      %28 = affine.apply #map1()[%0]
      %29 = arith.subi %28, %c1 : index
      %30 = polygeist.submap(%arg3, %29) {map = #map2} : (memref<?x20xf64>, index) -> memref<?xf64>
      %31 = polygeist.submap(%arg5, %29) {map = #map4} : (memref<?x20xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%30 : memref<?xf64>) outs(%31 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      }
      affine.for %arg7 = 1 to #map1()[%0] {
        affine.for %arg8 = 1 to #map1()[%0] {
          %76 = affine.load %arg4[%arg7, %arg8 - 1] : memref<?x20xf64>
          %77 = arith.mulf %10, %76 : f64
          %78 = arith.addf %77, %11 : f64
          %79 = arith.divf %16, %78 : f64
          affine.store %79, %arg4[%arg7, %arg8] : memref<?x20xf64>
          %80 = affine.load %arg2[%arg8, %arg7 - 1] : memref<?x20xf64>
          %81 = arith.mulf %17, %80 : f64
          %82 = affine.load %arg2[%arg8, %arg7] : memref<?x20xf64>
          %83 = arith.mulf %19, %82 : f64
          %84 = arith.addf %81, %83 : f64
          %85 = affine.load %arg2[%arg8, %arg7 + 1] : memref<?x20xf64>
          %86 = arith.mulf %13, %85 : f64
          %87 = arith.subf %84, %86 : f64
          %88 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x20xf64>
          %89 = arith.mulf %10, %88 : f64
          %90 = arith.subf %87, %89 : f64
          %91 = arith.divf %90, %78 : f64
          affine.store %91, %arg5[%arg7, %arg8] : memref<?x20xf64>
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      %32 = affine.apply #map1()[%0]
      %33 = arith.subi %32, %c1 : index
      %34 = polygeist.submap(%arg3, %0, %33) {map = #map5} : (memref<?x20xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%34 : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %35 = affine.apply #map1()[%0]
      %36 = arith.subi %35, %c1 : index
      %37 = affine.apply #map1()[%0]
      %38 = arith.subi %37, %c1 : index
      %39 = polygeist.submap(%arg4, %0, %38, %36) {map = #map6} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %40 = affine.apply #map1()[%0]
      %41 = arith.subi %40, %c1 : index
      %42 = polygeist.submap(%arg3, %0, %41, %36) {map = #map7} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %43 = affine.apply #map1()[%0]
      %44 = arith.subi %43, %c1 : index
      %45 = polygeist.submap(%arg5, %0, %44, %36) {map = #map6} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %46 = affine.apply #map1()[%0]
      %47 = arith.subi %46, %c1 : index
      %48 = polygeist.submap(%arg3, %0, %47, %36) {map = #map8} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map9, #map9, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%39, %42, %45 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%48 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64):
        %76 = arith.mulf %in, %in_2 : f64
        %77 = arith.addf %76, %in_3 : f64
        linalg.yield %77 : f64
      }
      %49 = affine.apply #map1()[%0]
      %50 = arith.subi %49, %c1 : index
      %51 = polygeist.submap(%arg2, %50) {map = #map4} : (memref<?x20xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%51 : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %52 = affine.apply #map1()[%0]
      %53 = arith.subi %52, %c1 : index
      %54 = polygeist.submap(%arg4, %53) {map = #map4} : (memref<?x20xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%54 : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst_1 : f64
      }
      %55 = affine.apply #map1()[%0]
      %56 = arith.subi %55, %c1 : index
      %57 = polygeist.submap(%arg2, %56) {map = #map4} : (memref<?x20xf64>, index) -> memref<?xf64>
      %58 = polygeist.submap(%arg5, %56) {map = #map4} : (memref<?x20xf64>, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%57 : memref<?xf64>) outs(%58 : memref<?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      }
      affine.for %arg7 = 1 to #map1()[%0] {
        affine.for %arg8 = 1 to #map1()[%0] {
          %76 = affine.load %arg4[%arg7, %arg8 - 1] : memref<?x20xf64>
          %77 = arith.mulf %13, %76 : f64
          %78 = arith.addf %77, %14 : f64
          %79 = arith.divf %17, %78 : f64
          affine.store %79, %arg4[%arg7, %arg8] : memref<?x20xf64>
          %80 = affine.load %arg3[%arg7 - 1, %arg8] : memref<?x20xf64>
          %81 = arith.mulf %16, %80 : f64
          %82 = affine.load %arg3[%arg7, %arg8] : memref<?x20xf64>
          %83 = arith.mulf %21, %82 : f64
          %84 = arith.addf %81, %83 : f64
          %85 = affine.load %arg3[%arg7 + 1, %arg8] : memref<?x20xf64>
          %86 = arith.mulf %10, %85 : f64
          %87 = arith.subf %84, %86 : f64
          %88 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x20xf64>
          %89 = arith.mulf %13, %88 : f64
          %90 = arith.subf %87, %89 : f64
          %91 = arith.divf %90, %78 : f64
          affine.store %91, %arg5[%arg7, %arg8] : memref<?x20xf64>
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
      %59 = affine.apply #map1()[%0]
      %60 = arith.subi %59, %c1 : index
      %61 = polygeist.submap(%arg2, %0, %60) {map = #map10} : (memref<?x20xf64>, index, index) -> memref<?xf64>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%61 : memref<?xf64>) {
      ^bb0(%out: f64):
        linalg.yield %cst : f64
      }
      %62 = affine.apply #map1()[%0]
      %63 = arith.subi %62, %c1 : index
      %64 = affine.apply #map1()[%0]
      %65 = arith.subi %64, %c1 : index
      %66 = polygeist.submap(%arg4, %0, %65, %63) {map = #map6} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %67 = affine.apply #map1()[%0]
      %68 = arith.subi %67, %c1 : index
      %69 = polygeist.submap(%arg2, %0, %68, %63) {map = #map11} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %70 = affine.apply #map1()[%0]
      %71 = arith.subi %70, %c1 : index
      %72 = polygeist.submap(%arg5, %0, %71, %63) {map = #map6} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      %73 = affine.apply #map1()[%0]
      %74 = arith.subi %73, %c1 : index
      %75 = polygeist.submap(%arg2, %0, %74, %63) {map = #map6} : (memref<?x20xf64>, index, index, index) -> memref<?x?xf64>
      linalg.generic {indexing_maps = [#map9, #map9, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%66, %69, %72 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%75 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_2: f64, %in_3: f64, %out: f64):
        %76 = arith.mulf %in, %in_2 : f64
        %77 = arith.addf %76, %in_3 : f64
        linalg.yield %77 : f64
      }
    }
    return
  }
}

