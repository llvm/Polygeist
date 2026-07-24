#map = affine_map<(d0, d1)[s0, s1] -> (d1 * 25 + d0 * 50 + s0 * 100 + s1)>
#map1 = affine_map<(d0, d1)[s0, s1] -> (d1 * 50 + s0 * 100 + s1 + d0 * 25)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
          affine.for %arg8 = 0 to 2 {
            %22 = arith.index_cast %arg8 : index to i32
            %alloca_2 = memref.alloca() : memref<f64>
            affine.store %cst_0, %alloca_2[] : memref<f64>
            %subview = memref.subview %alloca_1[%arg7, 0] [1, %c2] [1, 1] : memref<2x2xf64> to memref<?xf64, strided<[1], offset: ?>>
            %23 = polygeist.submap(%arg4, %arg5, %arg6, %c2, %c2) {map = #map} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
            %24 = polygeist.submap(%arg4, %arg5, %arg6, %c2, %c2) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
            %subview_3 = memref.subview %alloca_2[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_4 = memref.subview %alloca_1[%arg7, 0] [1, %c2] [1, 1] : memref<2x2xf64> to memref<?xf64, strided<[1], offset: ?>>
            %cast = memref.cast %subview_4 : memref<?xf64, strided<[1], offset: ?>> to memref<?xf64>
            %subview_5 = memref.subview %cast[0] [%c2] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
            linalg.generic {indexing_maps = [#map2, #map3, #map4, #map4, #map5], iterator_types = ["reduction", "reduction"]} ins(%subview_5, %subview, %23, %24 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1], offset: ?>>, memref<?x?xf64>, memref<?x?xf64>) outs(%subview_3 : memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
              %32 = linalg.index 0 : index
              %33 = arith.index_cast %32 : index to i32
              %34 = arith.cmpi eq, %33, %22 : i32
              %35 = arith.extui %34 : i1 to i32
              %36 = arith.sitofp %35 : i32 to f64
              %37 = linalg.index 1 : index
              %38 = arith.index_cast %37 : index to i32
              %39 = arith.mulf %36, %in_6 : f64
              %40 = arith.cmpi eq, %38, %22 : i32
              %41 = arith.extui %40 : i1 to i32
              %42 = arith.sitofp %41 : i32 to f64
              %43 = arith.mulf %42, %in : f64
              %44 = arith.addf %39, %43 : f64
              %45 = arith.addf %in_7, %in_8 : f64
              %46 = arith.mulf %44, %45 : f64
              %47 = arith.addf %out, %46 : f64
              linalg.yield %47 : f64
            }
            %25 = affine.load %alloca_2[] : memref<f64>
            %26 = affine.load %alloca_1[%arg7, %arg8] : memref<2x2xf64>
            %27 = arith.mulf %18, %26 : f64
            %28 = arith.mulf %27, %15 : f64
            %29 = arith.mulf %20, %25 : f64
            %30 = arith.addf %28, %29 : f64
            %31 = arith.mulf %17, %30 : f64
            affine.store %31, %alloca[%arg7, %arg8] : memref<2x2xf64>
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
        %21 = polygeist.submap(%arg4, %arg5, %arg6, %c2, %c2) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%alloca : memref<2x2xf64>) outs(%21 : memref<?x?xf64>) {
        ^bb0(%in: f64, %out: f64):
          linalg.yield %in : f64
        }
      }
    }
    return
  }
}
