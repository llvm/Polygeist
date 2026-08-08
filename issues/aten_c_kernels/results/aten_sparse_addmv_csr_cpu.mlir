module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_addmv_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg5 = 0 to 64 {
      %0 = affine.load %arg0[%arg5] : memref<?xi32>
      %1 = affine.load %arg0[%arg5 + 1] : memref<?xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.index_cast %0 : i32 to index
      %4 = scf.for %arg6 = %3 to %2 step %c1 iter_args(%arg7 = %cst) -> (f32) {
        %5 = memref.load %arg2[%arg6] : memref<?xf32>
        %6 = memref.load %arg1[%arg6] : memref<?xi32>
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg3[%7] : memref<?xf32>
        %9 = arith.mulf %5, %8 : f32
        %10 = arith.addf %arg7, %9 : f32
        scf.yield %10 : f32
      }
      affine.store %4, %arg4[%arg5] : memref<?xf32>
    }
    return
  }
}
