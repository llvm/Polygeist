module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nll_loss2d_backward_cpu(%arg0: memref<?x16x16xf32>, %arg1: memref<?x16x16xi32>, %arg2: memref<?xf32>, %arg3: memref<?x8x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "polygeist.memref2pointer"(%arg3) : (memref<?x8x16x16xf32>) -> !llvm.ptr
    affine.for %arg4 = 0 to 8192 {
      %1 = arith.index_cast %arg4 : index to i32
      %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %2 : f32, !llvm.ptr
    }
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 16 {
        affine.for %arg6 = 0 to 16 {
          %1 = affine.load %arg1[%arg4, %arg5, %arg6] : memref<?x16x16xi32>
          %2 = arith.index_cast %1 : i32 to index
          %3 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<?x16x16xf32>
          %4 = arith.negf %3 : f32
          %5 = memref.load %arg2[%2] : memref<?xf32>
          %6 = arith.mulf %4, %5 : f32
          memref.store %6, %arg3[%arg4, %2, %arg5, %arg6] : memref<?x8x16x16xf32>
        }
      }
    }
    return
  }
}
