#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_rms_norm(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.280000e+02 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst_1, %alloca[] : memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<?xf32>) outs(%alloca : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.mulf %in, %in : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    }
    %0 = affine.load %alloca[] : memref<f32>
    %1 = arith.divf %0, %cst : f32
    %2 = arith.addf %1, %arg3 : f32
    %3 = math.sqrt %2 : f32
    %4 = arith.divf %cst_0, %3 : f32
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg1, %arg0 : memref<?xf32>, memref<?xf32>) outs(%arg2 : memref<?xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %4, %in_2 : f32
      %6 = arith.mulf %in, %5 : f32
      linalg.yield %6 : f32
    }
    return
  }
}

