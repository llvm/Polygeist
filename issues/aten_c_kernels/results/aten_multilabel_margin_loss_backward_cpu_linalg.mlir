#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multilabel_margin_loss_backward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x4xi32>, %arg2: memref<?xf32>, %arg3: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.600000e+01 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %0, %alloca[] : memref<f32>
    affine.for %arg4 = 0 to 16 {
      %1 = affine.load %alloca[] : memref<f32>
      %subview = memref.subview %arg3[%arg4, 0] [1, %c16] [1, 1] : memref<?x16xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      affine.store %1, %alloca[] : memref<f32>
      affine.for %arg5 = 0 to 4 {
        %2 = affine.load %arg1[%arg4, %arg5] : memref<?x4xi32>
        %3 = arith.index_cast %2 : i32 to index
        %alloca_2 = memref.alloca(%c16) : memref<?xi32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_2 : memref<?xi32>) {
        ^bb0(%out: i32):
          linalg.yield %c0_i32 : i32
        }
        %subview_3 = memref.subview %arg1[%arg4, 0] [1, %c4] [1, 1] : memref<?x4xi32> to memref<?xi32, strided<[1], offset: ?>>
        %subview_4 = memref.subview %alloca_2[0] [%c16] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_3 : memref<?xi32, strided<[1], offset: ?>>) outs(%subview_4 : memref<?xi32, strided<[1]>>) {
        ^bb0(%in: i32, %out: i32):
          %4 = linalg.index 0 : index
          %5 = arith.index_cast %4 : index to i32
          %6 = arith.cmpi eq, %in, %5 : i32
          %7 = arith.extui %6 : i1 to i32
          %8 = arith.ori %out, %7 : i32
          linalg.yield %8 : i32
        }
        affine.for %arg6 = 0 to 16 {
          %4 = affine.load %alloca[] : memref<f32>
          %5 = affine.load %alloca_2[%arg6] : memref<?xi32>
          %6 = arith.cmpi ne, %5, %c0_i32 : i32
          %7 = scf.if %6 -> (f32) {
            scf.yield %4 : f32
          } else {
            %8 = memref.load %arg0[%arg4, %3] : memref<?x16xf32>
            %9 = arith.subf %cst_0, %8 : f32
            %10 = affine.load %arg0[%arg4, %arg6] : memref<?x16xf32>
            %11 = arith.addf %9, %10 : f32
            %12 = arith.cmpf ogt, %11, %cst_1 : f32
            %13 = scf.if %12 -> (f32) {
              %14 = affine.load %arg2[%arg4] : memref<?xf32>
              %15 = arith.divf %14, %cst : f32
              %16 = affine.load %arg3[%arg4, %arg6] : memref<?x16xf32>
              %17 = arith.addf %16, %15 : f32
              affine.store %17, %arg3[%arg4, %arg6] : memref<?x16xf32>
              %18 = memref.load %arg3[%arg4, %3] : memref<?x16xf32>
              %19 = arith.subf %18, %15 : f32
              memref.store %19, %arg3[%arg4, %3] : memref<?x16xf32>
              scf.yield %15 : f32
            } else {
              scf.yield %4 : f32
            }
            scf.yield %13 : f32
          }
          affine.store %7, %alloca[] : memref<f32>
        }
      }
    }
    return
  }
}

