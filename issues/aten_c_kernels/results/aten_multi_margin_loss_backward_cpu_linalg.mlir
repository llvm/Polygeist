#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multi_margin_loss_backward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 1.600000e+01 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    %alloca = memref.alloca(%c32) : memref<?xf32>
    affine.for %arg7 = 0 to 32 {
      %1 = affine.load %arg1[%arg7] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      affine.store %cst, %alloca[%arg7] : memref<?xf32>
      %subview = memref.subview %arg6[%arg7, 0] [1, %c16] [1, 1] : memref<?x16xf32> to memref<?xf32, strided<[1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview : memref<?xf32, strided<[1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      affine.for %arg8 = 0 to 16 {
        %5 = affine.load %alloca[%arg7] : memref<?xf32>
        %6 = arith.index_cast %arg8 : index to i32
        %7 = arith.cmpi ne, %6, %1 : i32
        %8 = scf.if %7 -> (f32) {
          %9 = memref.load %arg0[%arg7, %2] : memref<?x16xf32>
          %10 = arith.subf %arg3, %9 : f32
          %11 = affine.load %arg0[%arg7, %arg8] : memref<?x16xf32>
          %12 = arith.addf %10, %11 : f32
          %13 = arith.cmpf ogt, %12, %cst : f32
          %14 = scf.if %13 -> (f32) {
            %15 = affine.load %arg5[%arg7] : memref<?xf32>
            %16 = memref.load %arg2[%2] : memref<?xf32>
            %17 = arith.mulf %15, %16 : f32
            %18 = affine.apply #map1()[%0]
            %19 = arith.cmpi eq, %18, %c0 : index
            %20 = arith.mulf %12, %cst_1 : f32
            %21 = arith.select %19, %cst_0, %20 : f32
            %22 = arith.mulf %17, %21 : f32
            %23 = arith.divf %22, %cst_2 : f32
            affine.store %23, %arg6[%arg7, %arg8] : memref<?x16xf32>
            %24 = arith.addf %5, %23 : f32
            scf.yield %24 : f32
          } else {
            scf.yield %5 : f32
          }
          scf.yield %14 : f32
        } else {
          scf.yield %5 : f32
        }
        affine.store %8, %alloca[%arg7] : memref<?xf32>
      }
      %3 = affine.load %alloca[%arg7] : memref<?xf32>
      %4 = arith.negf %3 : f32
      memref.store %4, %arg6[%arg7, %2] : memref<?x16xf32>
    }
    return
  }
}

