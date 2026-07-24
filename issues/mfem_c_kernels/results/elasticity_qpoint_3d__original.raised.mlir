#map = affine_map<(d0, d1)[s0, s1] -> (d1 * 125 + d0 * 375 + s0 * 1125 + s1)>
#map1 = affine_map<(d0, d1)[s0, s1] -> (d1 * 375 + s0 * 1125 + s1 + d0 * 125)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
          affine.for %arg8 = 0 to 3 {
            %62 = arith.index_cast %arg8 : index to i32
            %alloca_2 = memref.alloca() : memref<f64>
            affine.store %cst_0, %alloca_2[] : memref<f64>
            %subview = memref.subview %alloca_1[%arg7, 0] [1, %c3] [1, 1] : memref<3x3xf64> to memref<?xf64, strided<[1], offset: ?>>
            %63 = polygeist.submap(%arg4, %arg5, %arg6, %c3, %c3) {map = #map} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
            %64 = polygeist.submap(%arg4, %arg5, %arg6, %c3, %c3) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
            %subview_3 = memref.subview %alloca_2[] [] [] : memref<f64> to memref<f64, strided<[]>>
            %subview_4 = memref.subview %alloca_1[%arg7, 0] [1, %c3] [1, 1] : memref<3x3xf64> to memref<?xf64, strided<[1], offset: ?>>
            %cast = memref.cast %subview_4 : memref<?xf64, strided<[1], offset: ?>> to memref<?xf64>
            %subview_5 = memref.subview %cast[0] [%c3] [1] : memref<?xf64> to memref<?xf64, strided<[1]>>
            linalg.generic {indexing_maps = [#map2, #map3, #map4, #map4, #map5], iterator_types = ["reduction", "reduction"]} ins(%subview_5, %subview, %63, %64 : memref<?xf64, strided<[1]>>, memref<?xf64, strided<[1], offset: ?>>, memref<?x?xf64>, memref<?x?xf64>) outs(%subview_3 : memref<f64, strided<[]>>) {
            ^bb0(%in: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
              %72 = linalg.index 0 : index
              %73 = arith.index_cast %72 : index to i32
              %74 = arith.cmpi eq, %73, %62 : i32
              %75 = arith.extui %74 : i1 to i32
              %76 = arith.sitofp %75 : i32 to f64
              %77 = linalg.index 1 : index
              %78 = arith.index_cast %77 : index to i32
              %79 = arith.mulf %76, %in_6 : f64
              %80 = arith.cmpi eq, %78, %62 : i32
              %81 = arith.extui %80 : i1 to i32
              %82 = arith.sitofp %81 : i32 to f64
              %83 = arith.mulf %82, %in : f64
              %84 = arith.addf %79, %83 : f64
              %85 = arith.addf %in_7, %in_8 : f64
              %86 = arith.mulf %84, %85 : f64
              %87 = arith.addf %out, %86 : f64
              linalg.yield %87 : f64
            }
            %65 = affine.load %alloca_2[] : memref<f64>
            %66 = affine.load %alloca_1[%arg7, %arg8] : memref<3x3xf64>
            %67 = arith.mulf %58, %66 : f64
            %68 = arith.mulf %67, %55 : f64
            %69 = arith.mulf %60, %65 : f64
            %70 = arith.addf %68, %69 : f64
            %71 = arith.mulf %57, %70 : f64
            affine.store %71, %alloca[%arg7, %arg8] : memref<3x3xf64>
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
        %61 = polygeist.submap(%arg4, %arg5, %arg6, %c3, %c3) {map = #map1} : (memref<?xf64>, index, index, index, index) -> memref<?x?xf64>
        linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%alloca : memref<3x3xf64>) outs(%61 : memref<?x?xf64>) {
        ^bb0(%in: f64, %out: f64):
          linalg.yield %in : f64
        }
      }
    }
    return
  }
}
