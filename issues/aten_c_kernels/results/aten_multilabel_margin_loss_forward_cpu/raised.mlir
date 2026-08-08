#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multilabel_margin_loss_forward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x4xi32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.600000e+01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    %alloca_2 = memref.alloca(%c16) : memref<?xf32>
    %alloca_3 = memref.alloca(%c16) : memref<?xf32>
    affine.for %arg3 = 0 to 16 {
      %1 = affine.load %alloca[] : memref<f32>
      affine.store %1, %alloca_2[%arg3] : memref<?xf32>
      affine.store %cst_1, %alloca_3[%arg3] : memref<?xf32>
      %alloca_4 = memref.alloca(%c4) : memref<?xf32>
      %alloca_5 = memref.alloca(%c4) : memref<?xf32>
      affine.for %arg4 = 0 to 4 {
        %5 = affine.load %alloca_2[%arg3] : memref<?xf32>
        %6 = affine.load %alloca_3[%arg3] : memref<?xf32>
        %7 = affine.load %arg1[%arg3, %arg4] : memref<?x4xi32>
        %8 = arith.index_cast %7 : i32 to index
        affine.store %5, %alloca_4[%arg4] : memref<?xf32>
        affine.store %6, %alloca_5[%arg4] : memref<?xf32>
        %alloca_6 = memref.alloca(%c16) : memref<?xi32>
        affine.for %arg5 = 0 to 16 {
          %11 = affine.load %alloca_4[%arg4] : memref<?xf32>
          %12 = affine.load %alloca_5[%arg4] : memref<?xf32>
          %13 = arith.index_cast %arg5 : index to i32
          affine.store %c0_i32, %alloca_6[%arg5] : memref<?xi32>
          %subview = memref.subview %arg1[%arg3, 0] [1, %c4] [1, 1] : memref<?x4xi32> to memref<?xi32, strided<[1], offset: ?>>
          %subview_7 = memref.subview %alloca_6[%arg5] [1] [1] : memref<?xi32> to memref<i32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xi32, strided<[1], offset: ?>>) outs(%subview_7 : memref<i32, strided<[], offset: ?>>) {
          ^bb0(%in: i32, %out: i32):
            %25 = arith.cmpi eq, %in, %13 : i32
            %26 = arith.extui %25 : i1 to i32
            %27 = arith.ori %out, %26 : i32
            linalg.yield %27 : i32
          }
          %14 = affine.load %alloca_6[%arg5] : memref<?xi32>
          %15 = arith.cmpi eq, %14, %c0_i32 : i32
          %16 = memref.load %arg0[%arg3, %8] : memref<?x16xf32>
          %17 = arith.subf %cst_0, %16 : f32
          %18 = affine.load %arg0[%arg3, %arg5] : memref<?x16xf32>
          %19 = arith.addf %17, %18 : f32
          %20 = arith.cmpf ogt, %19, %cst_1 : f32
          %21 = arith.addf %12, %19 : f32
          %22 = arith.select %20, %21, %12 : f32
          %23 = arith.select %15, %19, %11 : f32
          %24 = arith.select %15, %22, %12 : f32
          affine.store %23, %alloca_4[%arg4] : memref<?xf32>
          affine.store %24, %alloca_5[%arg4] : memref<?xf32>
        }
        %9 = affine.load %alloca_4[%arg4] : memref<?xf32>
        %10 = affine.load %alloca_5[%arg4] : memref<?xf32>
        affine.store %9, %alloca_2[%arg3] : memref<?xf32>
        affine.store %10, %alloca_3[%arg3] : memref<?xf32>
      }
      %2 = affine.load %alloca_2[%arg3] : memref<?xf32>
      %3 = affine.load %alloca_3[%arg3] : memref<?xf32>
      %4 = arith.divf %3, %cst : f32
      affine.store %4, %arg2[%arg3] : memref<?xf32>
      affine.store %2, %alloca[] : memref<f32>
    }
    return
  }
}

