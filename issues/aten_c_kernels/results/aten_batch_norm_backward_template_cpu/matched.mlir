#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_batch_norm_backward_template_cpu(%arg0: memref<?x16x16x16xf32>, %arg1: memref<?x16x16x16xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x16x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 2.048000e+03 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca(%c16) : memref<?xf32>
    %alloca_1 = memref.alloca(%c16) : memref<?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    affine.for %arg5 = 0 to 16 {
      %alloca_8 = memref.alloca(%c8) : memref<?xf32>
      %alloca_9 = memref.alloca(%c8) : memref<?xf32>
      affine.for %arg6 = 0 to 8 {
        %2 = affine.load %alloca[%arg5] : memref<?xf32>
        %3 = affine.load %alloca_1[%arg5] : memref<?xf32>
        %4 = affine.load %arg2[%arg5] : memref<?xf32>
        affine.store %2, %alloca_8[%arg6] : memref<?xf32>
        affine.store %3, %alloca_9[%arg6] : memref<?xf32>
        %alloca_10 = memref.alloca(%c16) : memref<?xf32>
        %alloca_11 = memref.alloca(%c16) : memref<?xf32>
        affine.for %arg7 = 0 to 16 {
          %7 = affine.load %alloca_8[%arg6] : memref<?xf32>
          %8 = affine.load %alloca_9[%arg6] : memref<?xf32>
          affine.store %7, %alloca_10[%arg7] : memref<?xf32>
          affine.store %8, %alloca_11[%arg7] : memref<?xf32>
          %subview_12 = memref.subview %arg0[%arg6, %arg5, %arg7, 0] [1, 1, 1, %c16] [1, 1, 1, 1] : memref<?x16x16x16xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_13 = memref.subview %arg1[%arg6, %arg5, %arg7, 0] [1, 1, 1, %c16] [1, 1, 1, 1] : memref<?x16x16x16xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_14 = memref.subview %alloca_10[%arg7] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          %subview_15 = memref.subview %alloca_11[%arg7] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["reduction"]} ins(%subview_12, %subview_13 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_14, %subview_15 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %in_16: f32, %out: f32, %out_17: f32):
            %11 = arith.addf %out_17, %in : f32
            %12 = arith.subf %in_16, %4 : f32
            %13 = arith.mulf %in, %12 : f32
            %14 = arith.addf %out, %13 : f32
            linalg.yield %14, %11 : f32, f32
          }
          %9 = affine.load %alloca_10[%arg7] : memref<?xf32>
          %10 = affine.load %alloca_11[%arg7] : memref<?xf32>
          affine.store %9, %alloca_8[%arg6] : memref<?xf32>
          affine.store %10, %alloca_9[%arg6] : memref<?xf32>
        }
        %5 = affine.load %alloca_8[%arg6] : memref<?xf32>
        %6 = affine.load %alloca_9[%arg6] : memref<?xf32>
        affine.store %5, %alloca[%arg5] : memref<?xf32>
        affine.store %6, %alloca_1[%arg5] : memref<?xf32>
      }
    } {polygeist.was_parallel}
    %subview = memref.subview %arg3[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_2 = memref.subview %arg0[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : memref<?x16x16x16xf32> to memref<?x?x?x?xf32, strided<[4096, 256, 16, 1]>>
    %subview_3 = memref.subview %arg1[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : memref<?x16x16x16xf32> to memref<?x?x?x?xf32, strided<[4096, 256, 16, 1]>>
    %subview_4 = memref.subview %arg2[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_5 = memref.subview %arg4[0, 0, 0, 0] [%c8, %c16, %c16, %c16] [1, 1, 1, 1] : memref<?x16x16x16xf32> to memref<?x?x?x?xf32, strided<[4096, 256, 16, 1]>>
    %0 = polygeist.submap(%alloca, %c16) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %subview_6 = memref.subview %0[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %1 = polygeist.submap(%alloca_1, %c16) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %subview_7 = memref.subview %1[0] [%c16] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map3, #map3, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_6, %subview_7, %subview, %subview_2, %subview_3, %subview_4 : memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>, memref<?x?x?x?xf32, strided<[4096, 256, 16, 1]>>, memref<?x?x?x?xf32, strided<[4096, 256, 16, 1]>>, memref<?xf32, strided<[1]>>) outs(%subview_5 : memref<?x?x?x?xf32, strided<[4096, 256, 16, 1]>>) {
    ^bb0(%in: f32, %in_8: f32, %in_9: f32, %in_10: f32, %in_11: f32, %in_12: f32, %out: f32):
      %2 = arith.divf %in_8, %cst : f32
      %3 = arith.subf %in_10, %2 : f32
      %4 = arith.subf %in_11, %in_12 : f32
      %5 = arith.mulf %4, %in_9 : f32
      %6 = arith.mulf %5, %in_9 : f32
      %7 = arith.mulf %6, %in : f32
      %8 = arith.divf %7, %cst : f32
      %9 = arith.subf %3, %8 : f32
      %10 = arith.mulf %in_9, %9 : f32
      linalg.yield %10 : f32
    }
    return
  }
}

