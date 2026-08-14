#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fractional_max_pool3d_cpu(%arg0: memref<?x2x8x9x10xf32>, %arg1: memref<?x2x3xf32>, %arg2: memref<?x2x3x4x5xf32>, %arg3: memref<?x2x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 6.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %cst_3 = arith.constant 4.000000e+00 : f32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_4 = arith.constant -3.40282347E+38 : f32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 3 {
        %0 = arith.index_cast %arg5 : index to i32
        %1 = arith.sitofp %0 : i32 to f32
        affine.for %arg6 = 0 to 4 {
          %2 = arith.index_cast %arg6 : index to i32
          %3 = arith.sitofp %2 : i32 to f32
          %alloca = memref.alloca(%c5) : memref<?xi32>
          %alloca_5 = memref.alloca(%c5) : memref<?xf32>
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
          ^bb0(%out: i32):
            linalg.yield %c0_i32 : i32
          }
          linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_5 : memref<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_4 : f32
          }
          affine.for %arg7 = 0 to 5 {
            %4 = arith.index_cast %arg7 : index to i32
            %5 = affine.load %arg1[0, %arg4, 0] : memref<?x2x3xf32>
            %6 = arith.addf %1, %5 : f32
            %7 = arith.mulf %6, %cst_0 : f32
            %8 = arith.divf %7, %cst : f32
            %9 = arith.fptosi %8 : f32 to i32
            %10 = affine.load %arg1[0, %arg4, 1] : memref<?x2x3xf32>
            %11 = arith.addf %3, %10 : f32
            %12 = arith.mulf %11, %cst_0 : f32
            %13 = arith.divf %12, %cst_1 : f32
            %14 = arith.fptosi %13 : f32 to i32
            %15 = arith.sitofp %4 : i32 to f32
            %16 = affine.load %arg1[0, %arg4, 2] : memref<?x2x3xf32>
            %17 = arith.addf %15, %16 : f32
            %18 = arith.mulf %17, %cst_2 : f32
            %19 = arith.divf %18, %cst_3 : f32
            %20 = arith.fptosi %19 : f32 to i32
            %21 = arith.cmpi sgt, %9, %c6_i32 : i32
            %22 = arith.select %21, %c6_i32, %9 : i32
            %23 = arith.cmpi sgt, %14, %c6_i32 : i32
            %24 = arith.select %23, %c6_i32, %14 : i32
            %25 = arith.cmpi sgt, %20, %c7_i32 : i32
            %26 = arith.select %25, %c7_i32, %20 : i32
            %alloca_9 = memref.alloca(%c2) : memref<?xi32>
            %alloca_10 = memref.alloca(%c2) : memref<?xf32>
            affine.for %arg8 = 0 to 2 {
              %27 = affine.load %alloca[%arg7] : memref<?xi32>
              %28 = affine.load %alloca_5[%arg7] : memref<?xf32>
              %29 = arith.index_cast %arg8 : index to i32
              %30 = arith.addi %22, %29 : i32
              %31 = arith.index_cast %30 : i32 to index
              %32 = arith.muli %30, %c9_i32 : i32
              %33 = arith.addi %32, %24 : i32
              affine.store %27, %alloca_9[%arg8] : memref<?xi32>
              affine.store %28, %alloca_10[%arg8] : memref<?xf32>
              %alloca_11 = memref.alloca(%c3) : memref<?xi32>
              %alloca_12 = memref.alloca(%c3) : memref<?xf32>
              affine.for %arg9 = 0 to 3 {
                %36 = affine.load %alloca_9[%arg8] : memref<?xi32>
                %37 = affine.load %alloca_10[%arg8] : memref<?xf32>
                %38 = arith.index_cast %arg9 : index to i32
                %39 = arith.addi %24, %38 : i32
                %40 = arith.index_cast %39 : i32 to index
                %41 = arith.addi %33, %38 : i32
                %42 = arith.muli %41, %c10_i32 : i32
                %43 = arith.addi %42, %26 : i32
                affine.store %36, %alloca_11[%arg9] : memref<?xi32>
                affine.store %37, %alloca_12[%arg9] : memref<?xf32>
                affine.for %arg10 = 0 to 3 {
                  %46 = affine.load %alloca_11[%arg9] : memref<?xi32>
                  %47 = affine.load %alloca_12[%arg9] : memref<?xf32>
                  %48 = arith.index_cast %arg10 : index to i32
                  %49 = arith.addi %26, %48 : i32
                  %50 = arith.index_cast %49 : i32 to index
                  %51 = memref.load %arg0[%c0, %arg4, %31, %40, %50] : memref<?x2x8x9x10xf32>
                  %52 = arith.cmpf ogt, %51, %47 : f32
                  %53 = memref.load %arg0[%c0, %arg4, %31, %40, %50] : memref<?x2x8x9x10xf32>
                  %54 = arith.addi %43, %48 : i32
                  %55 = arith.select %52, %54, %46 : i32
                  %56 = arith.select %52, %53, %47 : f32
                  affine.store %55, %alloca_11[%arg9] : memref<?xi32>
                  affine.store %56, %alloca_12[%arg9] : memref<?xf32>
                }
                %44 = affine.load %alloca_11[%arg9] : memref<?xi32>
                %45 = affine.load %alloca_12[%arg9] : memref<?xf32>
                affine.store %44, %alloca_9[%arg8] : memref<?xi32>
                affine.store %45, %alloca_10[%arg8] : memref<?xf32>
              }
              %34 = affine.load %alloca_9[%arg8] : memref<?xi32>
              %35 = affine.load %alloca_10[%arg8] : memref<?xf32>
              affine.store %34, %alloca[%arg7] : memref<?xi32>
              affine.store %35, %alloca_5[%arg7] : memref<?xf32>
            }
          }
          %subview = memref.subview %alloca_5[0] [%c5] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
          %subview_6 = memref.subview %arg2[0, %arg4, %arg5, %arg6, 0] [1, 1, 1, 1, %c5] [1, 1, 1, 1, 1] : memref<?x2x3x4x5xf32> to memref<?xf32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[1]>>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          %subview_7 = memref.subview %alloca[0] [%c5] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
          %subview_8 = memref.subview %arg3[0, %arg4, %arg5, %arg6, 0] [1, 1, 1, 1, %c5] [1, 1, 1, 1, 1] : memref<?x2x3x4x5xi32> to memref<?xi32, strided<[1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview_7 : memref<?xi32, strided<[1]>>) outs(%subview_8 : memref<?xi32, strided<[1], offset: ?>>) {
          ^bb0(%in: i32, %out: i32):
            linalg.yield %in : i32
          }
        }
      }
    }
    return
  }
}

