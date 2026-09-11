#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flash_attention_cpu(%arg0: memref<?x2x16x32xf32>, %arg1: memref<?x2x16x32xf32>, %arg2: memref<?x2x16x32xf32>, %arg3: memref<?x2x16x32xf32>, %arg4: memref<?x2x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.176776692 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %alloca = memref.alloca() : memref<16xf32>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 16 {
        %alloca_2 = memref.alloca() : memref<f32>
        affine.store %cst_1, %alloca_2[] : memref<f32>
        %alloca_3 = memref.alloca(%c16) : memref<?xf32>
        affine.for %arg7 = 0 to 16 {
          %4 = affine.load %alloca_2[] : memref<f32>
          affine.store %cst_0, %alloca_3[%arg7] : memref<?xf32>
          %subview_10 = memref.subview %arg0[0, %arg5, %arg6, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_11 = memref.subview %arg1[0, %arg5, %arg7, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_12 = memref.subview %alloca_3[%arg7] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"]} ins(%subview_10, %subview_11 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_12 : memref<f32, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %9 = arith.mulf %in, %in_13 : f32
            %10 = arith.addf %out, %9 : f32
            linalg.yield %10 : f32
          }
          %5 = affine.load %alloca_3[%arg7] : memref<?xf32>
          %6 = arith.mulf %5, %cst : f32
          affine.store %6, %alloca[%arg7] : memref<16xf32>
          %7 = arith.cmpf ogt, %6, %4 : f32
          %8 = arith.select %7, %6, %4 : f32
          affine.store %8, %alloca_2[] : memref<f32>
        }
        %0 = affine.load %alloca_2[] : memref<f32>
        %alloca_4 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_4[] : memref<f32>
        %subview = memref.subview %alloca[0] [%c16] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
        %subview_5 = memref.subview %alloca_4[] [] [] : memref<f32> to memref<f32, strided<[]>>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} outs(%subview, %subview_5 : memref<?xf32, strided<[1]>>, memref<f32, strided<[]>>) {
        ^bb0(%out: f32, %out_10: f32):
          %4 = arith.subf %out, %0 : f32
          %5 = math.exp %4 : f32
          %6 = arith.addf %out_10, %5 : f32
          linalg.yield %5, %6 : f32, f32
        }
        %1 = affine.load %alloca_4[] : memref<f32>
        %2 = math.log %1 : f32
        %3 = arith.addf %0, %2 : f32
        affine.store %3, %arg4[0, %arg5, %arg6] : memref<?x2x16xf32>
        %subview_6 = memref.subview %arg3[0, %arg5, %arg6, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_0 : f32
        }
        %subview_7 = memref.subview %alloca[0] [%c16] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
        %subview_8 = memref.subview %arg2[0, %arg5, 0, 0] [1, 1, %c16, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?x?xf32, strided<[32, 1], offset: ?>>
        %subview_9 = memref.subview %arg3[0, %arg5, %arg6, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%subview_7, %subview_8 : memref<?xf32, strided<[1]>>, memref<?x?xf32, strided<[32, 1], offset: ?>>) outs(%subview_9 : memref<?xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %in_10: f32, %out: f32):
          %4 = arith.divf %in, %1 : f32
          %5 = arith.mulf %4, %in_10 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
        }
      }
    }
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

