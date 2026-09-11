#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<?x64xf32>, %arg4: memref<?x64xf32>, %arg5: memref<?x64xf32>, %arg6: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant -2.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %1, %alloca[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    affine.store %1, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    affine.store %1, %alloca_4[] : memref<f32>
    %2 = arith.negf %arg2 : f32
    %3 = math.exp %2 : f32
    %4 = arith.subf %cst, %3 : f32
    %5 = arith.mulf %4, %4 : f32
    %6 = arith.mulf %arg2, %cst_0 : f32
    %7 = arith.mulf %6, %3 : f32
    %8 = arith.addf %7, %cst : f32
    %9 = math.exp %6 : f32
    %10 = arith.subf %8, %9 : f32
    %11 = arith.divf %5, %10 : f32
    %12 = arith.mulf %11, %3 : f32
    %13 = arith.subf %arg2, %cst : f32
    %14 = arith.mulf %12, %13 : f32
    %15 = math.powf %cst_0, %2 : f32
    %16 = arith.mulf %arg2, %cst_1 : f32
    %17 = math.exp %16 : f32
    %18 = arith.negf %17 : f32
    %19 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %19 {
      affine.store %cst_2, %alloca_3[] : memref<f32>
      affine.store %cst_2, %alloca[] : memref<f32>
      affine.store %cst_2, %alloca_4[] : memref<f32>
      %20 = polygeist.submap(%arg3, %arg7, %0) {map = #map} : (memref<?x64xf32>, index, index) -> memref<?xf32>
      %21 = polygeist.submap(%arg3, %arg7, %0) {map = #map} : (memref<?x64xf32>, index, index) -> memref<?xf32>
      %22 = polygeist.submap(%arg5, %arg7, %0) {map = #map} : (memref<?x64xf32>, index, index) -> memref<?xf32>
      %23 = polygeist.submap(%arg5, %arg7, %0) {map = #map} : (memref<?x64xf32>, index, index) -> memref<?xf32>
      %24 = polygeist.submap(%alloca_4, %0) {map = #map1} : (memref<f32>, index) -> memref<?xf32>
      %25 = polygeist.submap(%alloca, %0) {map = #map1} : (memref<f32>, index) -> memref<?xf32>
      %26 = polygeist.submap(%alloca_3, %0) {map = #map1} : (memref<f32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2, #map2, #map2], iterator_types = ["reduction"]} ins(%20, %21, %22 : memref<?xf32>, memref<?xf32>, memref<?xf32>) outs(%23, %24, %25, %26 : memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) {
      ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32, %out_7: f32, %out_8: f32, %out_9: f32):
        %27 = arith.mulf %11, %in : f32
        %28 = arith.mulf %14, %out_7 : f32
        %29 = arith.addf %27, %28 : f32
        %30 = arith.mulf %15, %out_9 : f32
        %31 = arith.addf %29, %30 : f32
        %32 = arith.mulf %18, %out_8 : f32
        %33 = arith.addf %31, %32 : f32
        linalg.yield %33, %in_5, %out_9, %in_6 : f32, f32, f32, f32
      }
    }
    return
  }
}

