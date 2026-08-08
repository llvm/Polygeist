module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_backward_other_arg_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?x24xi32>, %arg3: memref<?x24xf32>, %arg4: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg5 = 0 to 32 {
      affine.for %arg6 = 0 to 24 {
        affine.store %cst, %arg4[%arg5, %arg6] : memref<?x24xf32>
      }
    }
    affine.for %arg5 = 0 to 16 {
      affine.for %arg6 = 0 to 24 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<?x24xi32>
        %1 = arith.cmpi sge, %0, %c0_i32 : i32
        scf.if %1 {
          %2 = arith.index_cast %0 : i32 to index
          %3 = memref.load %arg0[%2] : memref<?xi32>
          %4 = arith.index_cast %3 : i32 to index
          %5 = memref.load %arg1[%2] : memref<?xf32>
          %6 = affine.load %arg3[%arg5, %arg6] : memref<?x24xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = memref.load %arg4[%4, %arg6] : memref<?x24xf32>
          %9 = arith.addf %8, %7 : f32
          memref.store %9, %arg4[%4, %arg6] : memref<?x24xf32>
        }
      }
    }
    return
  }
}
