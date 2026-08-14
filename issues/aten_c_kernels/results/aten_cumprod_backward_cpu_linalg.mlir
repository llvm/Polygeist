#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cumprod_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg3 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    affine.for %arg4 = 0 to 128 {
      %0 = arith.index_cast %arg4 : index to i32
      affine.for %arg5 = #map(%arg4) to 128 {
        %1 = affine.load %arg3[%arg4] : memref<?xf32>
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst, %alloca[] : memref<f32>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<?xf32>) outs(%alloca : memref<f32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = linalg.index 0 : index
          %7 = arith.index_cast %6 : index to i32
          %8 = arith.cmpi ne, %7, %0 : i32
          %9 = arith.mulf %out, %in : f32
          %10 = arith.select %8, %9, %out : f32
          %11 = linalg.index 0 : index
          %12 = affine.apply #map2(%arg5)
          %13 = arith.cmpi slt, %11, %12 : index
          %14 = arith.select %13, %10, %out : f32
          linalg.yield %14 : f32
        }
        %2 = affine.load %alloca[] : memref<f32>
        %3 = affine.load %arg2[%arg5] : memref<?xf32>
        %4 = arith.mulf %3, %2 : f32
        %5 = arith.addf %1, %4 : f32
        affine.store %5, %arg3[%arg4] : memref<?xf32>
      }
    } {polygeist.was_parallel}
    return
  }
}

