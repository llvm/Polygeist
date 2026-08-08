#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_topk_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<16x64xi32>
    %alloca_0 = memref.alloca() : memref<16x64xf32>
    %subview = memref.subview %arg0[0, 0] [%c16, %c64] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_1 = memref.subview %alloca_0[0, 0] [%c16, %c64] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<?x?xf32, strided<[64, 1]>>) outs(%subview_1 : memref<?x?xf32, strided<[64, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_2 = memref.subview %alloca[0, 0] [%c16, %c64] [1, 1] : memref<16x64xi32> to memref<?x?xi32, strided<[64, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview_2 : memref<?x?xi32, strided<[64, 1]>>) {
    ^bb0(%out: i32):
      %0 = linalg.index 1 : index
      %1 = arith.index_cast %0 : index to i32
      linalg.yield %1 : i32
    }
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 1 to 64 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = affine.load %alloca_0[%arg3, %arg4] : memref<16x64xf32>
        %2 = affine.load %alloca[%arg3, %arg4] : memref<16x64xi32>
        %3 = arith.addi %0, %c-1_i32 : i32
        %4 = scf.while (%arg5 = %3) : (i32) -> i32 {
          %7 = arith.cmpi sge, %arg5, %c0_i32 : i32
          %8:2 = scf.if %7 -> (i1, i32) {
            %9 = arith.index_cast %arg5 : i32 to index
            %10 = memref.load %alloca_0[%arg3, %9] : memref<16x64xf32>
            %11 = arith.cmpf olt, %10, %1 : f32
            %12 = scf.if %11 -> (i32) {
              %13 = arith.addi %arg5, %c1_i32 : i32
              %14 = arith.index_cast %13 : i32 to index
              memref.store %10, %alloca_0[%arg3, %14] : memref<16x64xf32>
              %15 = memref.load %alloca[%arg3, %9] : memref<16x64xi32>
              memref.store %15, %alloca[%arg3, %14] : memref<16x64xi32>
              %16 = arith.addi %arg5, %c-1_i32 : i32
              scf.yield %16 : i32
            } else {
              scf.yield %arg5 : i32
            }
            scf.yield %11, %12 : i1, i32
          } else {
            scf.yield %false, %arg5 : i1, i32
          }
          scf.condition(%8#0) %8#1 : i32
        } do {
        ^bb0(%arg5: i32):
          scf.yield %arg5 : i32
        }
        %5 = arith.addi %4, %c1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        memref.store %1, %alloca_0[%arg3, %6] : memref<16x64xf32>
        memref.store %2, %alloca[%arg3, %6] : memref<16x64xi32>
      }
    }
    %subview_3 = memref.subview %alloca_0[0, 0] [%c16, %c8] [1, 1] : memref<16x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    %subview_4 = memref.subview %arg1[0, 0] [%c16, %c8] [1, 1] : memref<?x8xf32> to memref<?x?xf32, strided<[8, 1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_3 : memref<?x?xf32, strided<[64, 1]>>) outs(%subview_4 : memref<?x?xf32, strided<[8, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_5 = memref.subview %alloca[0, 0] [%c16, %c8] [1, 1] : memref<16x64xi32> to memref<?x?xi32, strided<[64, 1]>>
    %subview_6 = memref.subview %arg2[0, 0] [%c16, %c8] [1, 1] : memref<?x8xi32> to memref<?x?xi32, strided<[8, 1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview_5 : memref<?x?xi32, strided<[64, 1]>>) outs(%subview_6 : memref<?x?xi32, strided<[8, 1]>>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

