#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flash_attention_backward_cpu(%arg0: memref<?x2x16x32xf32>, %arg1: memref<?x2x16x32xf32>, %arg2: memref<?x2x16x32xf32>, %arg3: memref<?x2x16x32xf32>, %arg4: memref<?x2x16x32xf32>, %arg5: memref<?x2x16x32xf32>, %arg6: memref<?x2x16x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.176776692 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<16xf32>
    %alloca_2 = memref.alloca() : memref<16xf32>
    %0 = "polygeist.memref2pointer"(%arg4) : (memref<?x2x16x32xf32>) -> !llvm.ptr
    affine.for %arg7 = 0 to 1024 {
      %3 = arith.index_cast %arg7 : index to i32
      %4 = llvm.getelementptr %0[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_1, %4 : f32, !llvm.ptr
    }
    %1 = "polygeist.memref2pointer"(%arg5) : (memref<?x2x16x32xf32>) -> !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg6) : (memref<?x2x16x32xf32>) -> !llvm.ptr
    affine.for %arg7 = 0 to 1024 {
      %3 = arith.index_cast %arg7 : index to i32
      %4 = llvm.getelementptr %1[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_1, %4 : f32, !llvm.ptr
      %5 = llvm.getelementptr %2[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_1, %5 : f32, !llvm.ptr
    }
    affine.for %arg7 = 0 to 2 {
      %alloca_3 = memref.alloca(%c16) : memref<?xf32>
      %alloca_4 = memref.alloca(%c16) : memref<?xf32>
      %alloca_5 = memref.alloca(%c16) : memref<?xf32>
      affine.for %arg8 = 0 to 16 {
        affine.store %cst_0, %alloca_3[%arg8] : memref<?xf32>
        %alloca_6 = memref.alloca(%c16) : memref<?xf32>
        affine.for %arg9 = 0 to 16 {
          %6 = affine.load %alloca_3[%arg8] : memref<?xf32>
          affine.store %cst_1, %alloca_6[%arg9] : memref<?xf32>
          %subview_9 = memref.subview %arg0[0, %arg7, %arg8, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_10 = memref.subview %arg1[0, %arg7, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_11 = memref.subview %alloca_6[%arg9] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"]} ins(%subview_9, %subview_10 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_11 : memref<f32, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %in_12: f32, %out: f32):
            %11 = arith.mulf %in, %in_12 : f32
            %12 = arith.addf %out, %11 : f32
            linalg.yield %12 : f32
          }
          %7 = affine.load %alloca_6[%arg9] : memref<?xf32>
          %8 = arith.mulf %7, %cst : f32
          affine.store %8, %alloca_2[%arg9] : memref<16xf32>
          %9 = arith.cmpf ogt, %8, %6 : f32
          %10 = arith.select %9, %8, %6 : f32
          affine.store %10, %alloca_3[%arg8] : memref<?xf32>
        }
        %3 = affine.load %alloca_3[%arg8] : memref<?xf32>
        affine.store %cst_1, %alloca_4[%arg8] : memref<?xf32>
        %subview = memref.subview %alloca_2[0] [%c16] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
        %subview_7 = memref.subview %alloca_4[%arg8] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} outs(%subview, %subview_7 : memref<?xf32, strided<[1]>>, memref<f32, strided<[], offset: ?>>) {
        ^bb0(%out: f32, %out_9: f32):
          %6 = arith.subf %out, %3 : f32
          %7 = math.exp %6 : f32
          %8 = arith.addf %out_9, %7 : f32
          linalg.yield %7, %8 : f32, f32
        }
        %4 = affine.load %alloca_4[%arg8] : memref<?xf32>
        affine.store %cst_1, %alloca_5[%arg8] : memref<?xf32>
        %alloca_8 = memref.alloca(%c16) : memref<?xf32>
        affine.for %arg9 = 0 to 16 {
          %6 = affine.load %alloca_5[%arg8] : memref<?xf32>
          %7 = affine.load %alloca_2[%arg9] : memref<16xf32>
          %8 = arith.divf %7, %4 : f32
          affine.store %8, %alloca_2[%arg9] : memref<16xf32>
          affine.store %cst_1, %alloca_8[%arg9] : memref<?xf32>
          %subview_9 = memref.subview %arg3[0, %arg7, %arg8, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_10 = memref.subview %arg2[0, %arg7, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_11 = memref.subview %arg6[0, %arg7, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_12 = memref.subview %alloca_8[%arg9] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map, #map1], iterator_types = ["reduction"]} ins(%subview_9, %subview_10 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%subview_11, %subview_12 : memref<?xf32, strided<[1], offset: ?>>, memref<f32, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32, %out_14: f32):
            %12 = arith.mulf %in, %in_13 : f32
            %13 = arith.addf %out_14, %12 : f32
            %14 = arith.mulf %8, %in : f32
            %15 = arith.addf %out, %14 : f32
            linalg.yield %15, %13 : f32, f32
          }
          %9 = affine.load %alloca_8[%arg9] : memref<?xf32>
          affine.store %9, %alloca[%arg9] : memref<16xf32>
          %10 = arith.mulf %9, %8 : f32
          %11 = arith.addf %6, %10 : f32
          affine.store %11, %alloca_5[%arg8] : memref<?xf32>
        }
        %5 = affine.load %alloca_5[%arg8] : memref<?xf32>
        affine.for %arg9 = 0 to 16 {
          %6 = affine.load %alloca_2[%arg9] : memref<16xf32>
          %7 = affine.load %alloca[%arg9] : memref<16xf32>
          %8 = arith.subf %7, %5 : f32
          %9 = arith.mulf %6, %8 : f32
          %10 = arith.mulf %9, %cst : f32
          %subview_9 = memref.subview %arg1[0, %arg7, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_10 = memref.subview %arg4[0, %arg7, %arg8, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview_9 : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_10 : memref<?xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            %11 = arith.mulf %10, %in : f32
            %12 = arith.addf %out, %11 : f32
            linalg.yield %12 : f32
          }
          %subview_11 = memref.subview %arg0[0, %arg7, %arg8, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          %subview_12 = memref.subview %arg5[0, %arg7, %arg9, 0] [1, 1, 1, %c32] [1, 1, 1, 1] : memref<?x2x16x32xf32> to memref<?xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview_11 : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_12 : memref<?xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            %11 = arith.mulf %10, %in : f32
            %12 = arith.addf %out, %11 : f32
            linalg.yield %12 : f32
          }
        }
      }
    }
    return
  }
}

