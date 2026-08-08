#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (-d0 + 9)>
#map4 = affine_map<(d0) -> (-d0 + 8)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ctc_loss_backward_cpu(%arg0: memref<?x4x12xf32>, %arg1: memref<?x5xi32>, %arg2: i32, %arg3: memref<?x24x11xf32>, %arg4: memref<?xf32>, %arg5: memref<?x4x12xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c24 = arith.constant 24 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c-2 = arith.constant -2 : index
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c22 = arith.constant 22 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<24x11xf32>
    %subview = memref.subview %arg0[0, 0, 0] [%c24, %c4, %c12] [1, 1, 1] : memref<?x4x12xf32> to memref<?x?x?xf32, strided<[48, 12, 1]>>
    %subview_1 = memref.subview %arg4[0] [%c4] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_2 = memref.subview %arg5[0, 0, 0] [%c24, %c4, %c12] [1, 1, 1] : memref<?x4x12xf32> to memref<?x?x?xf32, strided<[48, 12, 1]>>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview, %subview_1 : memref<?x?x?xf32, strided<[48, 12, 1]>>, memref<?xf32, strided<[1]>>) outs(%subview_2 : memref<?x?x?xf32, strided<[48, 12, 1]>>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %0 = math.exp %in : f32
      %1 = arith.mulf %0, %in_3 : f32
      linalg.yield %1 : f32
    }
    affine.for %arg6 = 0 to 4 {
      %subview_3 = memref.subview %alloca[0, 0] [%c24, %c11] [1, 1] : memref<24x11xf32> to memref<?x?xf32, strided<[11, 1]>>
      linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%subview_3 : memref<?x?xf32, strided<[11, 1]>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      affine.store %cst, %alloca[23, 9] : memref<24x11xf32>
      affine.store %cst, %alloca[23, 10] : memref<24x11xf32>
      affine.for %arg7 = 0 to 23 {
        %3 = arith.subi %c22, %arg7 : index
        %4 = arith.index_cast %3 : index to i32
        %5 = arith.addi %4, %c1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        affine.for %arg8 = 0 to 11 {
          %7 = arith.index_cast %arg8 : index to i32
          %8 = arith.andi %7, %c1_i32 : i32
          %9 = arith.cmpi ne, %8, %c0_i32 : i32
          %10 = arith.cmpi slt, %arg8, %c0 : index
          %11 = arith.subi %c-1, %arg8 : index
          %12 = arith.select %10, %11, %arg8 : index
          %13 = arith.divsi %12, %c2 : index
          %14 = arith.subi %c-1, %13 : index
          %15 = arith.select %10, %14, %13 : index
          %16 = memref.load %arg1[%arg6, %15] : memref<?x5xi32>
          %17 = arith.select %9, %16, %arg2 : i32
          %18 = affine.load %alloca[-%arg7 + 23, %arg8] : memref<24x11xf32>
          %19 = arith.index_cast %17 : i32 to index
          %20 = memref.load %arg0[%6, %arg6, %19] : memref<?x4x12xf32>
          %21 = math.exp %20 : f32
          %22 = arith.mulf %18, %21 : f32
          %23 = arith.addi %7, %c1_i32 : i32
          %24 = affine.apply #map3(%arg8)
          %25 = arith.cmpi sge, %24, %c0 : index
          %26 = arith.andi %23, %c1_i32 : i32
          %27 = arith.cmpi ne, %26, %c0_i32 : i32
          %28 = arith.addi %arg8, %c1 : index
          %29 = arith.cmpi slt, %28, %c0 : index
          %30 = arith.subi %c-2, %arg8 : index
          %31 = arith.select %29, %30, %28 : index
          %32 = arith.divsi %31, %c2 : index
          %33 = arith.subi %c-1, %32 : index
          %34 = arith.select %29, %33, %32 : index
          %35 = memref.load %arg1[%arg6, %34] : memref<?x5xi32>
          %36 = arith.select %27, %35, %arg2 : i32
          %37 = affine.load %alloca[-%arg7 + 23, %arg8 + 1] : memref<24x11xf32>
          %38 = arith.index_cast %36 : i32 to index
          %39 = memref.load %arg0[%6, %arg6, %38] : memref<?x4x12xf32>
          %40 = math.exp %39 : f32
          %41 = arith.mulf %37, %40 : f32
          %42 = arith.addf %22, %41 : f32
          %43 = arith.select %25, %42, %22 : f32
          %44 = arith.addi %7, %c2_i32 : i32
          %45 = affine.apply #map4(%arg8)
          %46 = arith.cmpi sge, %45, %c0 : index
          %47 = arith.andi %44, %c1_i32 : i32
          %48 = arith.cmpi ne, %47, %c0_i32 : i32
          %49 = arith.cmpi slt, %arg8, %c0 : index
          %50 = arith.subi %c-1, %arg8 : index
          %51 = arith.select %49, %50, %arg8 : index
          %52 = arith.divsi %51, %c2 : index
          %53 = arith.subi %c-1, %52 : index
          %54 = arith.select %49, %53, %52 : index
          %55 = arith.addi %54, %c1 : index
          %56 = memref.load %arg1[%arg6, %55] : memref<?x5xi32>
          %57 = arith.select %48, %56, %arg2 : i32
          %58 = arith.cmpi ne, %17, %arg2 : i32
          %59 = arith.cmpi ne, %17, %57 : i32
          %60 = arith.andi %58, %59 : i1
          %61 = affine.load %alloca[-%arg7 + 23, %arg8 + 2] : memref<24x11xf32>
          %62 = arith.index_cast %57 : i32 to index
          %63 = memref.load %arg0[%6, %arg6, %62] : memref<?x4x12xf32>
          %64 = math.exp %63 : f32
          %65 = arith.mulf %61, %64 : f32
          %66 = arith.addf %43, %65 : f32
          %67 = arith.select %60, %66, %43 : f32
          %68 = arith.select %46, %67, %43 : f32
          affine.store %68, %alloca[-%arg7 + 22, %arg8] : memref<24x11xf32>
        }
      }
      %0 = affine.load %arg3[%arg6, 23, 10] : memref<?x24x11xf32>
      %1 = affine.load %arg3[%arg6, 23, 9] : memref<?x24x11xf32>
      %2 = arith.addf %0, %1 : f32
      affine.for %arg7 = 0 to 24 {
        affine.for %arg8 = 0 to 11 {
          %3 = arith.index_cast %arg8 : index to i32
          %4 = arith.andi %3, %c1_i32 : i32
          %5 = arith.cmpi ne, %4, %c0_i32 : i32
          %6 = arith.cmpi slt, %arg8, %c0 : index
          %7 = arith.subi %c-1, %arg8 : index
          %8 = arith.select %6, %7, %arg8 : index
          %9 = arith.divsi %8, %c2 : index
          %10 = arith.subi %c-1, %9 : index
          %11 = arith.select %6, %10, %9 : index
          %12 = memref.load %arg1[%arg6, %11] : memref<?x5xi32>
          %13 = arith.select %5, %12, %arg2 : i32
          %14 = arith.index_cast %13 : i32 to index
          %15 = affine.load %arg4[%arg6] : memref<?xf32>
          %16 = affine.load %arg3[%arg6, %arg7, %arg8] : memref<?x24x11xf32>
          %17 = arith.mulf %15, %16 : f32
          %18 = affine.load %alloca[%arg7, %arg8] : memref<24x11xf32>
          %19 = arith.mulf %17, %18 : f32
          %20 = arith.divf %19, %2 : f32
          %21 = memref.load %arg5[%arg7, %arg6, %14] : memref<?x4x12xf32>
          %22 = arith.subf %21, %20 : f32
          memref.store %22, %arg5[%arg7, %arg6, %14] : memref<?x4x12xf32>
        }
      }
    }
    return
  }
}

