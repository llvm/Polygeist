#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ctc_loss_cpu(%arg0: memref<?x4x12xf32>, %arg1: memref<?x5xi32>, %arg2: i32, %arg3: memref<?xf32>, %arg4: memref<?x24x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c11 = arith.constant 11 : index
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c-2_i32 = arith.constant -2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %subview = memref.subview %arg4[0, 0, 0] [%c4, %c24, %c11] [1, 1, 1] : memref<?x24x11xf32> to memref<?x?x?xf32, strided<[264, 11, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf32, strided<[264, 11, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %subview_0 = memref.subview %arg0[0, 0, %0] [1, %c4, 1] [1, 1, 1] : memref<?x4x12xf32> to memref<?xf32, strided<[12], offset: ?>>
    %subview_1 = memref.subview %arg4[0, 0, 0] [%c4, 1, 1] [1, 1, 1] : memref<?x24x11xf32> to memref<?xf32, strided<[264]>>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%subview_0 : memref<?xf32, strided<[12], offset: ?>>) outs(%subview_1 : memref<?xf32, strided<[264]>>) {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp %in : f32
      linalg.yield %1 : f32
    }
    affine.for %arg5 = 0 to 4 {
      %1 = affine.load %arg1[%arg5, 0] : memref<?x5xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = memref.load %arg0[%c0, %arg5, %2] : memref<?x4x12xf32>
      %4 = math.exp %3 : f32
      affine.store %4, %arg4[%arg5, 0, 1] : memref<?x24x11xf32>
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 1 to 24 {
        affine.for %arg7 = 0 to 11 {
          %1 = arith.index_cast %arg7 : index to i32
          %2 = arith.andi %1, %c1_i32 : i32
          %3 = arith.cmpi ne, %2, %c0_i32 : i32
          %4 = arith.cmpi slt, %arg7, %c0 : index
          %5 = arith.subi %c-1, %arg7 : index
          %6 = arith.select %4, %5, %arg7 : index
          %7 = arith.divsi %6, %c2 : index
          %8 = arith.subi %c-1, %7 : index
          %9 = arith.select %4, %8, %7 : index
          %10 = memref.load %arg1[%arg5, %9] : memref<?x5xi32>
          %11 = arith.select %3, %10, %arg2 : i32
          %12 = affine.load %arg4[%arg5, %arg6 - 1, %arg7] : memref<?x24x11xf32>
          %13 = affine.apply #map2(%arg7)
          %14 = arith.cmpi sge, %13, %c0 : index
          %15 = affine.load %arg4[%arg5, %arg6 - 1, %arg7 - 1] : memref<?x24x11xf32>
          %16 = arith.addf %12, %15 : f32
          %17 = arith.select %14, %16, %12 : f32
          %18 = arith.cmpi sgt, %1, %c1_i32 : i32
          %19 = arith.cmpi ne, %11, %arg2 : i32
          %20 = arith.andi %18, %19 : i1
          %21 = arith.addi %1, %c-2_i32 : i32
          %22 = arith.andi %21, %c1_i32 : i32
          %23 = arith.cmpi ne, %22, %c0_i32 : i32
          %24 = arith.cmpi slt, %arg7, %c0 : index
          %25 = arith.subi %c-1, %arg7 : index
          %26 = arith.select %24, %25, %arg7 : index
          %27 = arith.divsi %26, %c2 : index
          %28 = arith.subi %c-1, %27 : index
          %29 = arith.select %24, %28, %27 : index
          %30 = arith.addi %29, %c-1 : index
          %31 = memref.load %arg1[%arg5, %30] : memref<?x5xi32>
          %32 = arith.select %23, %31, %arg2 : i32
          %33 = arith.cmpi ne, %11, %32 : i32
          %34 = affine.load %arg4[%arg5, %arg6 - 1, %arg7 - 2] : memref<?x24x11xf32>
          %35 = arith.addf %17, %34 : f32
          %36 = arith.select %33, %35, %17 : f32
          %37 = arith.select %20, %36, %17 : f32
          %38 = arith.index_cast %11 : i32 to index
          %39 = memref.load %arg0[%arg6, %arg5, %38] : memref<?x4x12xf32>
          %40 = math.exp %39 : f32
          %41 = arith.mulf %37, %40 : f32
          affine.store %41, %arg4[%arg5, %arg6, %arg7] : memref<?x24x11xf32>
        }
      }
    }
    %subview_2 = memref.subview %arg4[0, 23, 10] [%c4, 1, 1] [1, 1, 1] : memref<?x24x11xf32> to memref<?xf32, strided<[264], offset: 263>>
    %subview_3 = memref.subview %arg4[0, 23, 9] [%c4, 1, 1] [1, 1, 1] : memref<?x24x11xf32> to memref<?xf32, strided<[264], offset: 262>>
    %subview_4 = memref.subview %arg3[0] [%c4] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%subview_2, %subview_3 : memref<?xf32, strided<[264], offset: 263>>, memref<?xf32, strided<[264], offset: 262>>) outs(%subview_4 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %1 = arith.addf %in, %in_5 : f32
      %2 = math.log %1 : f32
      %3 = arith.negf %2 : f32
      linalg.yield %3 : f32
    }
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

