#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_backward_cpu(%arg0: memref<?x3x8x8xf32>, %arg1: memref<?x6x6x2xf32>, %arg2: memref<?x3x6x6xf32>, %arg3: memref<?x3x8x8xf32>, %arg4: memref<?x6x6x2xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = "polygeist.memref2pointer"(%arg3) : (memref<?x3x8x8xf32>) -> !llvm.ptr
    affine.for %arg5 = 0 to 192 {
      %1 = arith.index_cast %arg5 : index to i32
      %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_2, %2 : f32, !llvm.ptr
    }
    affine.for %arg5 = 0 to 6 {
      %alloca = memref.alloca(%c6) : memref<?xf32>
      %alloca_3 = memref.alloca(%c6) : memref<?xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      }
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_3 : memref<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      }
      affine.for %arg6 = 0 to 6 {
        %1 = affine.load %arg1[0, %arg5, %arg6, 0] : memref<?x6x6x2xf32>
        %2 = arith.addf %1, %cst_1 : f32
        %3 = arith.mulf %2, %cst_0 : f32
        %4 = arith.mulf %3, %cst : f32
        %5 = affine.load %arg1[0, %arg5, %arg6, 1] : memref<?x6x6x2xf32>
        %6 = arith.addf %5, %cst_1 : f32
        %7 = arith.mulf %6, %cst_0 : f32
        %8 = arith.mulf %7, %cst : f32
        %9 = arith.fptosi %4 : f32 to i32
        %10 = arith.fptosi %8 : f32 to i32
        %11 = arith.addi %9, %c1_i32 : i32
        %12 = arith.addi %10, %c1_i32 : i32
        %13 = arith.sitofp %9 : i32 to f32
        %14 = arith.subf %4, %13 : f32
        %15 = arith.sitofp %10 : i32 to f32
        %16 = arith.subf %8, %15 : f32
        %17 = arith.cmpi sge, %9, %c0_i32 : i32
        %18 = arith.cmpi slt, %9, %c8_i32 : i32
        %19 = arith.cmpi sge, %10, %c0_i32 : i32
        %20 = arith.cmpi slt, %10, %c8_i32 : i32
        %21 = arith.andi %19, %20 : i1
        %22 = arith.andi %18, %21 : i1
        %23 = arith.andi %17, %22 : i1
        %24 = arith.cmpi sge, %11, %c0_i32 : i32
        %25 = arith.cmpi slt, %11, %c8_i32 : i32
        %26 = arith.andi %25, %21 : i1
        %27 = arith.andi %24, %26 : i1
        %28 = arith.cmpi sge, %12, %c0_i32 : i32
        %29 = arith.cmpi slt, %12, %c8_i32 : i32
        %30 = arith.andi %28, %29 : i1
        %31 = arith.andi %18, %30 : i1
        %32 = arith.andi %17, %31 : i1
        %33 = arith.andi %25, %30 : i1
        %34 = arith.andi %24, %33 : i1
        %35 = arith.subf %cst_1, %16 : f32
        %36 = arith.subf %cst_1, %14 : f32
        %37 = arith.index_cast %10 : i32 to index
        %38 = arith.index_cast %9 : i32 to index
        %39 = arith.index_cast %11 : i32 to index
        %40 = arith.index_cast %12 : i32 to index
        affine.for %arg7 = 0 to 3 {
          %41 = affine.load %alloca[%arg6] : memref<?xf32>
          %42 = affine.load %alloca_3[%arg6] : memref<?xf32>
          %43 = affine.load %arg2[0, %arg7, %arg5, %arg6] : memref<?x3x6x6xf32>
          %44 = scf.if %23 -> (f32) {
            %62 = memref.load %arg0[%c0, %arg7, %37, %38] : memref<?x3x8x8xf32>
            %63 = arith.mulf %43, %36 : f32
            %64 = arith.mulf %63, %35 : f32
            %65 = memref.load %arg3[%c0, %arg7, %37, %38] : memref<?x3x8x8xf32>
            %66 = arith.addf %65, %64 : f32
            memref.store %66, %arg3[%c0, %arg7, %37, %38] : memref<?x3x8x8xf32>
            scf.yield %62 : f32
          } else {
            scf.yield %cst_2 : f32
          }
          %45 = scf.if %27 -> (f32) {
            %62 = memref.load %arg0[%c0, %arg7, %37, %39] : memref<?x3x8x8xf32>
            %63 = arith.mulf %43, %14 : f32
            %64 = arith.mulf %63, %35 : f32
            %65 = memref.load %arg3[%c0, %arg7, %37, %39] : memref<?x3x8x8xf32>
            %66 = arith.addf %65, %64 : f32
            memref.store %66, %arg3[%c0, %arg7, %37, %39] : memref<?x3x8x8xf32>
            scf.yield %62 : f32
          } else {
            scf.yield %cst_2 : f32
          }
          %46 = scf.if %32 -> (f32) {
            %62 = memref.load %arg0[%c0, %arg7, %40, %38] : memref<?x3x8x8xf32>
            %63 = arith.mulf %43, %36 : f32
            %64 = arith.mulf %63, %16 : f32
            %65 = memref.load %arg3[%c0, %arg7, %40, %38] : memref<?x3x8x8xf32>
            %66 = arith.addf %65, %64 : f32
            memref.store %66, %arg3[%c0, %arg7, %40, %38] : memref<?x3x8x8xf32>
            scf.yield %62 : f32
          } else {
            scf.yield %cst_2 : f32
          }
          %47 = scf.if %34 -> (f32) {
            %62 = memref.load %arg0[%c0, %arg7, %40, %39] : memref<?x3x8x8xf32>
            %63 = arith.mulf %43, %14 : f32
            %64 = arith.mulf %63, %16 : f32
            %65 = memref.load %arg3[%c0, %arg7, %40, %39] : memref<?x3x8x8xf32>
            %66 = arith.addf %65, %64 : f32
            memref.store %66, %arg3[%c0, %arg7, %40, %39] : memref<?x3x8x8xf32>
            scf.yield %62 : f32
          } else {
            scf.yield %cst_2 : f32
          }
          %48 = arith.subf %45, %44 : f32
          %49 = arith.mulf %48, %35 : f32
          %50 = arith.subf %47, %46 : f32
          %51 = arith.mulf %50, %16 : f32
          %52 = arith.addf %49, %51 : f32
          %53 = arith.mulf %43, %52 : f32
          %54 = arith.addf %42, %53 : f32
          %55 = arith.subf %46, %44 : f32
          %56 = arith.mulf %55, %36 : f32
          %57 = arith.subf %47, %45 : f32
          %58 = arith.mulf %57, %14 : f32
          %59 = arith.addf %56, %58 : f32
          %60 = arith.mulf %43, %59 : f32
          %61 = arith.addf %41, %60 : f32
          affine.store %61, %alloca[%arg6] : memref<?xf32>
          affine.store %54, %alloca_3[%arg6] : memref<?xf32>
        }
      }
      %subview = memref.subview %alloca_3[0] [%c6] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %subview_4 = memref.subview %arg4[0, %arg5, 0, 0] [1, 1, %c6, 1] [1, 1, 1, 1] : memref<?x6x6x2xf32> to memref<?xf32, strided<[2], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[1]>>) outs(%subview_4 : memref<?xf32, strided<[2], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.mulf %in, %cst_0 : f32
        %2 = arith.mulf %1, %cst : f32
        linalg.yield %2 : f32
      }
      %subview_5 = memref.subview %alloca[0] [%c6] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      %subview_6 = memref.subview %arg4[0, %arg5, 0, 1] [1, 1, %c6, 1] [1, 1, 1, 1] : memref<?x6x6x2xf32> to memref<?xf32, strided<[2], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview_5 : memref<?xf32, strided<[1]>>) outs(%subview_6 : memref<?xf32, strided<[2], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.mulf %in, %cst_0 : f32
        %2 = arith.mulf %1, %cst : f32
        linalg.yield %2 : f32
      }
    }
    return
  }
}

