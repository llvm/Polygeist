#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 2 + 2)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 56 + s1 * 336 + s2 * 8)>
#map4 = affine_map<(d0) -> ()>
#map5 = affine_map<(d0)[s0, s1, s2] -> (d0 + s0 * 36 + s1 * 12 + s2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_max_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 3 {
          %alloca = memref.alloca(%c4) : memref<?xi32>
          %alloca_0 = memref.alloca(%c4) : memref<?xf32>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
          ^bb0(%out: i32):
            linalg.yield %c0_i32 : i32
          }
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_0 : memref<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          }
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = #map1(%arg4) to #map2(%arg4) {
              %2 = affine.load %alloca[%arg6] : memref<?xi32>
              %3 = affine.load %alloca_0[%arg6] : memref<?xf32>
              %4 = arith.index_cast %arg7 : index to i32
              %5 = arith.muli %4, %c7_i32 : i32
              %alloca_1 = memref.alloca() : memref<i32>
              affine.store %2, %alloca_1[] : memref<i32>
              %alloca_2 = memref.alloca() : memref<f32>
              affine.store %3, %alloca_2[] : memref<f32>
              affine.for %arg8 = #map1(%arg5) to #map2(%arg5) {
                %8 = affine.load %alloca_1[] : memref<i32>
                %9 = affine.load %alloca_2[] : memref<f32>
                %10 = arith.index_cast %arg8 : index to i32
                %11 = arith.addi %5, %10 : i32
                %12 = arith.muli %11, %c8_i32 : i32
                %alloca_3 = memref.alloca() : memref<i32>
                affine.store %8, %alloca_3[] : memref<i32>
                %alloca_4 = memref.alloca() : memref<f32>
                affine.store %9, %alloca_4[] : memref<f32>
                %13 = polygeist.submap(%arg0, %arg7, %arg3, %arg8, %c8) {map = #map3} : (memref<?xf32>, index, index, index, index) -> memref<?xf32>
                linalg.generic {indexing_maps = [#map, #map4, #map4], iterator_types = ["reduction"]} ins(%13 : memref<?xf32>) outs(%alloca_3, %alloca_4 : memref<i32>, memref<f32>) {
                ^bb0(%in: f32, %out: i32, %out_5: f32):
                  %16 = linalg.index 0 : index
                  %17 = arith.index_cast %16 : index to i32
                  %18 = arith.cmpf ogt, %in, %out_5 : f32
                  %19 = arith.select %18, %in, %out_5 : f32
                  %20 = arith.addi %12, %17 : i32
                  %21 = arith.select %18, %20, %out : i32
                  %22 = linalg.index 0 : index
                  %23 = affine.apply #map1(%arg6)
                  %24 = arith.cmpi sge, %22, %23 : index
                  %25 = affine.apply #map2(%arg6)
                  %26 = arith.cmpi slt, %22, %25 : index
                  %27 = arith.andi %24, %26 : i1
                  %28 = arith.select %27, %21, %out : i32
                  %29 = arith.select %27, %19, %out_5 : f32
                  linalg.yield %28, %29 : i32, f32
                }
                %14 = affine.load %alloca_3[] : memref<i32>
                %15 = affine.load %alloca_4[] : memref<f32>
                affine.store %14, %alloca_1[] : memref<i32>
                affine.store %15, %alloca_2[] : memref<f32>
              }
              %6 = affine.load %alloca_1[] : memref<i32>
              %7 = affine.load %alloca_2[] : memref<f32>
              affine.store %6, %alloca[%arg6] : memref<?xi32>
              affine.store %7, %alloca_0[%arg6] : memref<?xf32>
            }
          } {polygeist.was_parallel}
          %0 = polygeist.submap(%arg1, %arg3, %arg4, %arg5, %c4) {map = #map5} : (memref<?xf32>, index, index, index, index) -> memref<?xf32>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca_0 : memref<?xf32>) outs(%0 : memref<?xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          %1 = polygeist.submap(%arg2, %arg3, %arg4, %arg5, %c4) {map = #map5} : (memref<?xi32>, index, index, index, index) -> memref<?xi32>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca : memref<?xi32>) outs(%1 : memref<?xi32>) {
          ^bb0(%in: i32, %out: i32):
            linalg.yield %in : i32
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

