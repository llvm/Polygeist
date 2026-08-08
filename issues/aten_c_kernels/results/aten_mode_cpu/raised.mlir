#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_mode_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %c63 = arith.constant 63 : index
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<64xi32>
    %alloca_0 = memref.alloca() : memref<64xf32>
    %subview = memref.subview %arg0[0] [%c64] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_1 = memref.subview %alloca_0[0] [%c64] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[1]>>) outs(%subview_1 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_2 = memref.subview %alloca[0] [%c64] [1] : memref<64xi32> to memref<?xi32, strided<[1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview_2 : memref<?xi32, strided<[1]>>) {
    ^bb0(%out: i32):
      %4 = linalg.index 0 : index
      %5 = arith.index_cast %4 : index to i32
      linalg.yield %5 : i32
    }
    affine.for %arg3 = 1 to 64 {
      %4 = arith.index_cast %arg3 : index to i32
      %5 = affine.load %alloca_0[%arg3] : memref<64xf32>
      %6 = affine.load %alloca[%arg3] : memref<64xi32>
      %7 = arith.addi %4, %c-1_i32 : i32
      %8 = scf.while (%arg4 = %7) : (i32) -> i32 {
        %11 = arith.cmpi sge, %arg4, %c0_i32 : i32
        %12:2 = scf.if %11 -> (i1, i32) {
          %13 = arith.index_cast %arg4 : i32 to index
          %14 = memref.load %alloca_0[%13] : memref<64xf32>
          %15 = arith.cmpf ogt, %14, %5 : f32
          %16 = scf.if %15 -> (i32) {
            %17 = arith.addi %arg4, %c1_i32 : i32
            %18 = arith.index_cast %17 : i32 to index
            memref.store %14, %alloca_0[%18] : memref<64xf32>
            %19 = memref.load %alloca[%13] : memref<64xi32>
            memref.store %19, %alloca[%18] : memref<64xi32>
            %20 = arith.addi %arg4, %c-1_i32 : i32
            scf.yield %20 : i32
          } else {
            scf.yield %arg4 : i32
          }
          scf.yield %15, %16 : i1, i32
        } else {
          scf.yield %false, %arg4 : i1, i32
        }
        scf.condition(%12#0) %12#1 : i32
      } do {
      ^bb0(%arg4: i32):
        scf.yield %arg4 : i32
      }
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = arith.index_cast %9 : i32 to index
      memref.store %5, %alloca_0[%10] : memref<64xf32>
      memref.store %6, %alloca[%10] : memref<64xi32>
    }
    %alloca_3 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_3[] : memref<i32>
    %alloca_4 = memref.alloca() : memref<i32>
    affine.store %c1_i32, %alloca_4[] : memref<i32>
    %alloca_5 = memref.alloca() : memref<i32>
    affine.store %c1_i32, %alloca_5[] : memref<i32>
    %subview_6 = memref.subview %alloca_0[1] [%c63] [1] : memref<64xf32> to memref<?xf32, strided<[1], offset: 1>>
    %subview_7 = memref.subview %alloca_0[0] [%c63] [1] : memref<64xf32> to memref<?xf32, strided<[1]>>
    %subview_8 = memref.subview %alloca_3[] [] [] : memref<i32> to memref<i32, strided<[]>>
    %subview_9 = memref.subview %alloca_4[] [] [] : memref<i32> to memref<i32, strided<[]>>
    %subview_10 = memref.subview %alloca_5[] [] [] : memref<i32> to memref<i32, strided<[]>>
    linalg.generic {indexing_maps = [#map, #map, #map1, #map1, #map1], iterator_types = ["reduction"]} ins(%subview_6, %subview_7 : memref<?xf32, strided<[1], offset: 1>>, memref<?xf32, strided<[1]>>) outs(%subview_8, %subview_9, %subview_10 : memref<i32, strided<[]>>, memref<i32, strided<[]>>, memref<i32, strided<[]>>) {
    ^bb0(%in: f32, %in_11: f32, %out: i32, %out_12: i32, %out_13: i32):
      %4 = linalg.index 0 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.cmpf oeq, %in, %in_11 : f32
      %7 = arith.addi %out_12, %c1_i32 : i32
      %8 = arith.select %6, %7, %c1_i32 : i32
      %9 = arith.cmpi sgt, %8, %out_13 : i32
      %10 = arith.select %9, %5, %out : i32
      %11 = arith.select %9, %8, %out_13 : i32
      linalg.yield %10, %8, %11 : i32, i32, i32
    }
    %0 = affine.load %alloca_3[] : memref<i32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = affine.load %alloca_0[symbol(%1)] : memref<64xf32>
    affine.store %2, %arg1[0] : memref<?xf32>
    %3 = affine.load %alloca[symbol(%1)] : memref<64xi32>
    affine.store %3, %arg2[0] : memref<?xi32>
    return
  }
}

