#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 20 + d0 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_aa_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.571428597 : f32
    %cst_0 = arith.constant 6.250000e-01 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %cst_2 = arith.constant 2.500000e+00 : f32
    %cst_3 = arith.constant 1.500000e+00 : f32
    %cst_4 = arith.constant 2.000000e+00 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %cst_6 = arith.constant 5.000000e-01 : f32
    %cst_7 = arith.constant 1.000000e+00 : f32
    %cst_8 = arith.constant 4.000000e+00 : f32
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 7 {
        %0 = arith.index_cast %arg3 : index to i32
        %1 = arith.sitofp %0 : i32 to f32
        %2 = arith.addf %1, %cst_6 : f32
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.subf %3, %cst_6 : f32
        affine.for %arg4 = 0 to 8 {
          %5 = arith.index_cast %arg4 : index to i32
          %6 = arith.sitofp %5 : i32 to f32
          %7 = arith.addf %6, %cst_6 : f32
          %8 = arith.mulf %7, %cst_0 : f32
          %9 = arith.subf %8, %cst_6 : f32
          %alloca = memref.alloca() : memref<f32>
          affine.store %cst_5, %alloca[] : memref<f32>
          %10 = polygeist.submap(%alloca, %c4, %c5) {map = #map} : (memref<f32>, index, index) -> memref<?x?xf32>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["reduction", "reduction"]} outs(%10 : memref<?x?xf32>) {
          ^bb0(%out: f32):
            %15 = linalg.index 0 : index
            %16 = arith.index_cast %15 : index to i32
            %17 = arith.sitofp %16 : i32 to f32
            %18 = arith.subf %4, %17 : f32
            %19 = arith.cmpf olt, %18, %cst_5 : f32
            %20 = arith.negf %18 : f32
            %21 = arith.select %19, %20, %18 : f32
            %22 = arith.cmpf olt, %21, %cst_4 : f32
            %23 = arith.cmpf olt, %21, %cst_7 : f32
            %24 = arith.mulf %21, %cst_3 : f32
            %25 = arith.subf %24, %cst_2 : f32
            %26 = arith.mulf %25, %21 : f32
            %27 = arith.mulf %26, %21 : f32
            %28 = arith.addf %27, %cst_7 : f32
            %29 = arith.mulf %21, %cst_1 : f32
            %30 = arith.addf %29, %cst_2 : f32
            %31 = arith.mulf %30, %21 : f32
            %32 = arith.subf %31, %cst_8 : f32
            %33 = arith.mulf %32, %21 : f32
            %34 = arith.addf %33, %cst_4 : f32
            %35 = arith.select %23, %28, %34 : f32
            %36 = arith.select %22, %35, %cst_5 : f32
            %37 = linalg.index 1 : index
            %38 = arith.index_cast %37 : index to i32
            %39 = arith.sitofp %38 : i32 to f32
            %40 = arith.subf %9, %39 : f32
            %41 = arith.cmpf olt, %40, %cst_5 : f32
            %42 = arith.negf %40 : f32
            %43 = arith.select %41, %42, %40 : f32
            %44 = arith.cmpf olt, %43, %cst_4 : f32
            %45 = arith.cmpf olt, %43, %cst_7 : f32
            %46 = arith.mulf %43, %cst_3 : f32
            %47 = arith.subf %46, %cst_2 : f32
            %48 = arith.mulf %47, %43 : f32
            %49 = arith.mulf %48, %43 : f32
            %50 = arith.addf %49, %cst_7 : f32
            %51 = arith.mulf %43, %cst_1 : f32
            %52 = arith.addf %51, %cst_2 : f32
            %53 = arith.mulf %52, %43 : f32
            %54 = arith.subf %53, %cst_8 : f32
            %55 = arith.mulf %54, %43 : f32
            %56 = arith.addf %55, %cst_4 : f32
            %57 = arith.select %45, %50, %56 : f32
            %58 = arith.select %44, %57, %cst_5 : f32
            %59 = arith.mulf %36, %58 : f32
            %60 = arith.addf %out, %59 : f32
            linalg.yield %60 : f32
          }
          %11 = affine.load %alloca[] : memref<f32>
          %alloca_9 = memref.alloca() : memref<f32>
          affine.store %cst_5, %alloca_9[] : memref<f32>
          %12 = polygeist.submap(%arg0, %arg2, %c4, %c5) {map = #map2} : (memref<?xf32>, index, index, index) -> memref<?x?xf32>
          linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["reduction", "reduction"]} ins(%12 : memref<?x?xf32>) outs(%alloca_9 : memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %15 = linalg.index 0 : index
            %16 = arith.index_cast %15 : index to i32
            %17 = arith.sitofp %16 : i32 to f32
            %18 = arith.subf %4, %17 : f32
            %19 = arith.cmpf olt, %18, %cst_5 : f32
            %20 = arith.negf %18 : f32
            %21 = arith.select %19, %20, %18 : f32
            %22 = arith.cmpf olt, %21, %cst_4 : f32
            %23 = arith.cmpf olt, %21, %cst_7 : f32
            %24 = arith.mulf %21, %cst_3 : f32
            %25 = arith.subf %24, %cst_2 : f32
            %26 = arith.mulf %25, %21 : f32
            %27 = arith.mulf %26, %21 : f32
            %28 = arith.addf %27, %cst_7 : f32
            %29 = arith.mulf %21, %cst_1 : f32
            %30 = arith.addf %29, %cst_2 : f32
            %31 = arith.mulf %30, %21 : f32
            %32 = arith.subf %31, %cst_8 : f32
            %33 = arith.mulf %32, %21 : f32
            %34 = arith.addf %33, %cst_4 : f32
            %35 = arith.select %23, %28, %34 : f32
            %36 = arith.select %22, %35, %cst_5 : f32
            %37 = linalg.index 1 : index
            %38 = arith.index_cast %37 : index to i32
            %39 = arith.sitofp %38 : i32 to f32
            %40 = arith.subf %9, %39 : f32
            %41 = arith.cmpf olt, %40, %cst_5 : f32
            %42 = arith.negf %40 : f32
            %43 = arith.select %41, %42, %40 : f32
            %44 = arith.cmpf olt, %43, %cst_4 : f32
            %45 = arith.cmpf olt, %43, %cst_7 : f32
            %46 = arith.mulf %43, %cst_3 : f32
            %47 = arith.subf %46, %cst_2 : f32
            %48 = arith.mulf %47, %43 : f32
            %49 = arith.mulf %48, %43 : f32
            %50 = arith.addf %49, %cst_7 : f32
            %51 = arith.mulf %43, %cst_1 : f32
            %52 = arith.addf %51, %cst_2 : f32
            %53 = arith.mulf %52, %43 : f32
            %54 = arith.subf %53, %cst_8 : f32
            %55 = arith.mulf %54, %43 : f32
            %56 = arith.addf %55, %cst_4 : f32
            %57 = arith.select %45, %50, %56 : f32
            %58 = arith.select %44, %57, %cst_5 : f32
            %59 = arith.mulf %in, %36 : f32
            %60 = arith.mulf %59, %58 : f32
            %61 = arith.addf %out, %60 : f32
            linalg.yield %61 : f32
          }
          %13 = affine.load %alloca_9[] : memref<f32>
          %14 = arith.divf %13, %11 : f32
          affine.store %14, %arg1[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

