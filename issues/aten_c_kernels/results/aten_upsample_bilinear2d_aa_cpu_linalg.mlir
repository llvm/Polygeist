#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 20 + d0 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d_aa_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.571428597 : f32
    %cst_0 = arith.constant 6.250000e-01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 7 {
        %0 = arith.index_cast %arg3 : index to i32
        %1 = arith.sitofp %0 : i32 to f32
        %2 = arith.addf %1, %cst_2 : f32
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.subf %3, %cst_2 : f32
        affine.for %arg4 = 0 to 8 {
          %5 = arith.index_cast %arg4 : index to i32
          %6 = arith.sitofp %5 : i32 to f32
          %7 = arith.addf %6, %cst_2 : f32
          %8 = arith.mulf %7, %cst_0 : f32
          %9 = arith.subf %8, %cst_2 : f32
          %alloca = memref.alloca() : memref<f32>
          affine.store %cst_1, %alloca[] : memref<f32>
          %10 = polygeist.submap(%alloca, %c4, %c5) {map = #map} : (memref<f32>, index, index) -> memref<?x?xf32>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["reduction", "reduction"]} outs(%10 : memref<?x?xf32>) {
          ^bb0(%out: f32):
            %15 = linalg.index 0 : index
            %16 = arith.index_cast %15 : index to i32
            %17 = arith.sitofp %16 : i32 to f32
            %18 = arith.subf %4, %17 : f32
            %19 = arith.cmpf olt, %18, %cst_1 : f32
            %20 = arith.negf %18 : f32
            %21 = arith.select %19, %20, %18 : f32
            %22 = arith.cmpf olt, %21, %cst_3 : f32
            %23 = arith.subf %cst_3, %21 : f32
            %24 = arith.select %22, %23, %cst_1 : f32
            %25 = linalg.index 1 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = arith.sitofp %26 : i32 to f32
            %28 = arith.subf %9, %27 : f32
            %29 = arith.cmpf olt, %28, %cst_1 : f32
            %30 = arith.negf %28 : f32
            %31 = arith.select %29, %30, %28 : f32
            %32 = arith.cmpf olt, %31, %cst_3 : f32
            %33 = arith.subf %cst_3, %31 : f32
            %34 = arith.select %32, %33, %cst_1 : f32
            %35 = arith.mulf %24, %34 : f32
            %36 = arith.addf %out, %35 : f32
            linalg.yield %36 : f32
          }
          %11 = affine.load %alloca[] : memref<f32>
          %alloca_4 = memref.alloca() : memref<f32>
          affine.store %cst_1, %alloca_4[] : memref<f32>
          %12 = polygeist.submap(%arg0, %arg2, %c4, %c5) {map = #map2} : (memref<?xf32>, index, index, index) -> memref<?x?xf32>
          linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["reduction", "reduction"]} ins(%12 : memref<?x?xf32>) outs(%alloca_4 : memref<f32>) {
          ^bb0(%in: f32, %out: f32):
            %15 = linalg.index 0 : index
            %16 = arith.index_cast %15 : index to i32
            %17 = arith.sitofp %16 : i32 to f32
            %18 = arith.subf %4, %17 : f32
            %19 = arith.cmpf olt, %18, %cst_1 : f32
            %20 = arith.negf %18 : f32
            %21 = arith.select %19, %20, %18 : f32
            %22 = arith.cmpf olt, %21, %cst_3 : f32
            %23 = arith.subf %cst_3, %21 : f32
            %24 = arith.select %22, %23, %cst_1 : f32
            %25 = linalg.index 1 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = arith.sitofp %26 : i32 to f32
            %28 = arith.subf %9, %27 : f32
            %29 = arith.cmpf olt, %28, %cst_1 : f32
            %30 = arith.negf %28 : f32
            %31 = arith.select %29, %30, %28 : f32
            %32 = arith.cmpf olt, %31, %cst_3 : f32
            %33 = arith.subf %cst_3, %31 : f32
            %34 = arith.select %32, %33, %cst_1 : f32
            %35 = arith.mulf %in, %24 : f32
            %36 = arith.mulf %35, %34 : f32
            %37 = arith.addf %out, %36 : f32
            linalg.yield %37 : f32
          }
          %13 = affine.load %alloca_4[] : memref<f32>
          %14 = arith.divf %13, %11 : f32
          affine.store %14, %arg1[%arg4 + %arg2 * 56 + %arg3 * 8] : memref<?xf32>
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

