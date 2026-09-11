#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_backward_other_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x24xf32>, %arg4: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c24 = arith.constant 24 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg4[0, 0] [%c32, %c24] [1, 1] : memref<?x24xf32> to memref<?x?xf32, strided<[24, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[24, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg5 = 0 to 16 {
      %0 = affine.load %arg0[%arg5] : memref<?xi32>
      %1 = scf.while (%arg6 = %0) : (i32) -> i32 {
        %2 = affine.load %arg0[%arg5 + 1] : memref<?xi32>
        %3 = arith.cmpi slt, %arg6, %2 : i32
        scf.condition(%3) %arg6 : i32
      } do {
      ^bb0(%arg6: i32):
        %2 = arith.index_cast %arg6 : i32 to index
        affine.for %arg7 = 0 to 24 {
          %4 = memref.load %arg1[%2] : memref<?xi32>
          %5 = arith.index_cast %4 : i32 to index
          %6 = memref.load %arg2[%2] : memref<?xf32>
          %7 = affine.load %arg3[%arg5, %arg7] : memref<?x24xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = memref.load %arg4[%5, %arg7] : memref<?x24xf32>
          %10 = arith.addf %9, %8 : f32
          memref.store %10, %arg4[%5, %arg7] : memref<?x24xf32>
        }
        %3 = arith.addi %arg6, %c1_i32 : i32
        scf.yield %3 : i32
      }
    }
    return
  }
}

