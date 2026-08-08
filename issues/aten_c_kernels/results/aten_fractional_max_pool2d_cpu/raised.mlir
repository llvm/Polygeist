#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fractional_max_pool2d_cpu(%arg0: memref<?x3x9x10xf32>, %arg1: memref<?x3x2xf32>, %arg2: memref<?x3x4x5xf32>, %arg3: memref<?x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e+00 : f32
    %cst_1 = arith.constant 7.000000e+00 : f32
    %cst_2 = arith.constant 4.000000e+00 : f32
    %c10_i32 = arith.constant 10 : i32
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 4 {
          %0 = arith.index_cast %arg6 : index to i32
          %1 = arith.sitofp %0 : i32 to f32
          %alloca = memref.alloca(%c5) : memref<?xi32>
          %alloca_4 = memref.alloca(%c5) : memref<?xf32>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
          ^bb0(%out: i32):
            linalg.yield %c0_i32 : i32
          }
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_4 : memref<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_3 : f32
          }
          affine.for %arg7 = 0 to 5 {
            %2 = arith.index_cast %arg7 : index to i32
            %3 = affine.load %arg1[%arg4, %arg5, 0] : memref<?x3x2xf32>
            %4 = arith.addf %1, %3 : f32
            %5 = arith.mulf %4, %cst : f32
            %6 = arith.divf %5, %cst_0 : f32
            %7 = arith.fptosi %6 : f32 to i32
            %8 = arith.sitofp %2 : i32 to f32
            %9 = affine.load %arg1[%arg4, %arg5, 1] : memref<?x3x2xf32>
            %10 = arith.addf %8, %9 : f32
            %11 = arith.mulf %10, %cst_1 : f32
            %12 = arith.divf %11, %cst_2 : f32
            %13 = arith.fptosi %12 : f32 to i32
            %14 = arith.cmpi sgt, %7, %c6_i32 : i32
            %15 = arith.select %14, %c6_i32, %7 : i32
            %16 = arith.cmpi sgt, %13, %c7_i32 : i32
            %17 = arith.select %16, %c7_i32, %13 : i32
            %alloca_8 = memref.alloca(%c3) : memref<?xi32>
            %alloca_9 = memref.alloca(%c3) : memref<?xf32>
            affine.for %arg8 = 0 to 3 {
              %18 = affine.load %alloca[%arg7] : memref<?xi32>
              %19 = affine.load %alloca_4[%arg7] : memref<?xf32>
              %20 = arith.index_cast %arg8 : index to i32
              %21 = arith.addi %15, %20 : i32
              %22 = arith.index_cast %21 : i32 to index
              %23 = arith.muli %21, %c10_i32 : i32
              %24 = arith.addi %23, %17 : i32
              affine.store %18, %alloca_8[%arg8] : memref<?xi32>
              affine.store %19, %alloca_9[%arg8] : memref<?xf32>
              affine.for %arg9 = 0 to 3 {
                %27 = affine.load %alloca_8[%arg8] : memref<?xi32>
                %28 = affine.load %alloca_9[%arg8] : memref<?xf32>
                %29 = arith.index_cast %arg9 : index to i32
                %30 = arith.addi %17, %29 : i32
                %31 = arith.index_cast %30 : i32 to index
                %32 = memref.load %arg0[%arg4, %arg5, %22, %31] : memref<?x3x9x10xf32>
                %33 = arith.cmpf ogt, %32, %28 : f32
                %34 = memref.load %arg0[%arg4, %arg5, %22, %31] : memref<?x3x9x10xf32>
                %35 = arith.addi %24, %29 : i32
                %36 = arith.select %33, %35, %27 : i32
                %37 = arith.select %33, %34, %28 : f32
                affine.store %36, %alloca_8[%arg8] : memref<?xi32>
                affine.store %37, %alloca_9[%arg8] : memref<?xf32>
              }
              %25 = affine.load %alloca_8[%arg8] : memref<?xi32>
              %26 = affine.load %alloca_9[%arg8] : memref<?xf32>
              affine.store %25, %alloca[%arg7] : memref<?xi32>
              affine.store %26, %alloca_4[%arg7] : memref<?xf32>
            }
          }
          %subview = memref.subview %alloca_4[0] [%c5] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
          %subview_5 = memref.subview %arg2[%arg4, %arg5, %arg6, 0] [1, 1, 1, %c5] [1, 1, 1, 1] : memref<?x3x4x5xf32> to memref<?xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[1]>>) outs(%subview_5 : memref<?xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          %subview_6 = memref.subview %alloca[0] [%c5] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
          %subview_7 = memref.subview %arg3[%arg4, %arg5, %arg6, 0] [1, 1, 1, %c5] [1, 1, 1, 1] : memref<?x3x4x5xi32> to memref<?xi32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview_6 : memref<?xi32, strided<[1]>>) outs(%subview_7 : memref<?xi32, strided<[1], offset: ?>>) {
          ^bb0(%in: i32, %out: i32):
            linalg.yield %in : i32
          }
        }
      }
    }
    return
  }
}

