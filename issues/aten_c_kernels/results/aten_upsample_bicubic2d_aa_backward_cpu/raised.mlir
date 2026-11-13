#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1)[s0, s1, s2] -> (s0 + s1 * 56 + s2 * 8)>
#map4 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 20 + d0 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_aa_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.571428597 : f32
    %cst_0 = arith.constant 6.250000e-01 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %cst_2 = arith.constant 2.500000e+00 : f32
    %cst_3 = arith.constant 1.500000e+00 : f32
    %cst_4 = arith.constant 2.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 4.000000e+00 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_8 : f32
    }
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 7 {
        %0 = arith.index_cast %arg3 : index to i32
        %1 = arith.sitofp %0 : i32 to f32
        %2 = arith.addf %1, %cst_5 : f32
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.subf %3, %cst_5 : f32
        affine.for %arg4 = 0 to 8 {
          %5 = arith.index_cast %arg4 : index to i32
          %6 = arith.sitofp %5 : i32 to f32
          %7 = arith.addf %6, %cst_5 : f32
          %8 = arith.mulf %7, %cst_0 : f32
          %9 = arith.subf %8, %cst_5 : f32
          %alloca = memref.alloca() : memref<f32>
          affine.store %cst_8, %alloca[] : memref<f32>
          %10 = polygeist.submap(%alloca, %c4, %c5) {map = #map1} : (memref<f32>, index, index) -> memref<?x?xf32>
          linalg.generic {indexing_maps = [#map2], iterator_types = ["reduction", "reduction"]} outs(%10 : memref<?x?xf32>) {
          ^bb0(%out: f32):
            %14 = linalg.index 0 : index
            %15 = arith.index_cast %14 : index to i32
            %16 = arith.sitofp %15 : i32 to f32
            %17 = arith.subf %4, %16 : f32
            %18 = arith.cmpf olt, %17, %cst_8 : f32
            %19 = arith.negf %17 : f32
            %20 = arith.select %18, %19, %17 : f32
            %21 = arith.cmpf olt, %20, %cst_4 : f32
            %22 = arith.cmpf olt, %20, %cst_6 : f32
            %23 = arith.mulf %20, %cst_3 : f32
            %24 = arith.subf %23, %cst_2 : f32
            %25 = arith.mulf %24, %20 : f32
            %26 = arith.mulf %25, %20 : f32
            %27 = arith.addf %26, %cst_6 : f32
            %28 = arith.mulf %20, %cst_1 : f32
            %29 = arith.addf %28, %cst_2 : f32
            %30 = arith.mulf %29, %20 : f32
            %31 = arith.subf %30, %cst_7 : f32
            %32 = arith.mulf %31, %20 : f32
            %33 = arith.addf %32, %cst_4 : f32
            %34 = arith.select %22, %27, %33 : f32
            %35 = arith.select %21, %34, %cst_8 : f32
            %36 = linalg.index 1 : index
            %37 = arith.index_cast %36 : index to i32
            %38 = arith.sitofp %37 : i32 to f32
            %39 = arith.subf %9, %38 : f32
            %40 = arith.cmpf olt, %39, %cst_8 : f32
            %41 = arith.negf %39 : f32
            %42 = arith.select %40, %41, %39 : f32
            %43 = arith.cmpf olt, %42, %cst_4 : f32
            %44 = arith.cmpf olt, %42, %cst_6 : f32
            %45 = arith.mulf %42, %cst_3 : f32
            %46 = arith.subf %45, %cst_2 : f32
            %47 = arith.mulf %46, %42 : f32
            %48 = arith.mulf %47, %42 : f32
            %49 = arith.addf %48, %cst_6 : f32
            %50 = arith.mulf %42, %cst_1 : f32
            %51 = arith.addf %50, %cst_2 : f32
            %52 = arith.mulf %51, %42 : f32
            %53 = arith.subf %52, %cst_7 : f32
            %54 = arith.mulf %53, %42 : f32
            %55 = arith.addf %54, %cst_4 : f32
            %56 = arith.select %44, %49, %55 : f32
            %57 = arith.select %43, %56, %cst_8 : f32
            %58 = arith.mulf %35, %57 : f32
            %59 = arith.addf %out, %58 : f32
            linalg.yield %59 : f32
          }
          %11 = affine.load %alloca[] : memref<f32>
          %12 = polygeist.submap(%arg0, %arg4, %arg2, %arg3, %c4, %c5) {map = #map3} : (memref<?xf32>, index, index, index, index, index) -> memref<?x?xf32>
          %13 = polygeist.submap(%arg1, %arg2, %c4, %c5) {map = #map4} : (memref<?xf32>, index, index, index) -> memref<?x?xf32>
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%12 : memref<?x?xf32>) outs(%13 : memref<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = linalg.index 0 : index
            %15 = arith.index_cast %14 : index to i32
            %16 = arith.sitofp %15 : i32 to f32
            %17 = arith.subf %4, %16 : f32
            %18 = arith.cmpf olt, %17, %cst_8 : f32
            %19 = arith.negf %17 : f32
            %20 = arith.select %18, %19, %17 : f32
            %21 = arith.cmpf olt, %20, %cst_4 : f32
            %22 = arith.cmpf olt, %20, %cst_6 : f32
            %23 = arith.mulf %20, %cst_3 : f32
            %24 = arith.subf %23, %cst_2 : f32
            %25 = arith.mulf %24, %20 : f32
            %26 = arith.mulf %25, %20 : f32
            %27 = arith.addf %26, %cst_6 : f32
            %28 = arith.mulf %20, %cst_1 : f32
            %29 = arith.addf %28, %cst_2 : f32
            %30 = arith.mulf %29, %20 : f32
            %31 = arith.subf %30, %cst_7 : f32
            %32 = arith.mulf %31, %20 : f32
            %33 = arith.addf %32, %cst_4 : f32
            %34 = arith.select %22, %27, %33 : f32
            %35 = arith.select %21, %34, %cst_8 : f32
            %36 = linalg.index 1 : index
            %37 = arith.index_cast %36 : index to i32
            %38 = arith.sitofp %37 : i32 to f32
            %39 = arith.subf %9, %38 : f32
            %40 = arith.cmpf olt, %39, %cst_8 : f32
            %41 = arith.negf %39 : f32
            %42 = arith.select %40, %41, %39 : f32
            %43 = arith.cmpf olt, %42, %cst_4 : f32
            %44 = arith.cmpf olt, %42, %cst_6 : f32
            %45 = arith.mulf %42, %cst_3 : f32
            %46 = arith.subf %45, %cst_2 : f32
            %47 = arith.mulf %46, %42 : f32
            %48 = arith.mulf %47, %42 : f32
            %49 = arith.addf %48, %cst_6 : f32
            %50 = arith.mulf %42, %cst_1 : f32
            %51 = arith.addf %50, %cst_2 : f32
            %52 = arith.mulf %51, %42 : f32
            %53 = arith.subf %52, %cst_7 : f32
            %54 = arith.mulf %53, %42 : f32
            %55 = arith.addf %54, %cst_4 : f32
            %56 = arith.select %44, %49, %55 : f32
            %57 = arith.select %43, %56, %cst_8 : f32
            %58 = arith.mulf %in, %35 : f32
            %59 = arith.mulf %58, %57 : f32
            %60 = arith.divf %59, %11 : f32
            %61 = arith.addf %out, %60 : f32
            linalg.yield %61 : f32
          }
        }
      }
    } {polygeist.was_parallel}
    return
  }
}

