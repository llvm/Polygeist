module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sampled_addmm_sparse_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x32xf32>, %arg4: memref<?x24xf32>, %arg5: f32, %arg6: f32, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg8 = 0 to 16 {
      %0 = affine.load %arg0[%arg8] : memref<?xi32>
      %1 = scf.while (%arg9 = %0) : (i32) -> i32 {
        %2 = affine.load %arg0[%arg8 + 1] : memref<?xi32>
        %3 = arith.cmpi slt, %arg9, %2 : i32
        scf.condition(%3) %arg9 : i32
      } do {
      ^bb0(%arg9: i32):
        %2 = arith.index_cast %arg9 : i32 to index
        %3 = memref.load %arg1[%2] : memref<?xi32>
        %4 = arith.index_cast %3 : i32 to index
        %5 = affine.for %arg10 = 0 to 32 iter_args(%arg11 = %cst) -> (f32) {
          %11 = affine.load %arg3[%arg8, %arg10] : memref<?x32xf32>
          %12 = memref.load %arg4[%arg10, %4] : memref<?x24xf32>
          %13 = arith.mulf %11, %12 : f32
          %14 = arith.addf %arg11, %13 : f32
          affine.yield %14 : f32
        }
        %6 = memref.load %arg2[%2] : memref<?xf32>
        %7 = arith.mulf %arg6, %6 : f32
        %8 = arith.mulf %arg5, %5 : f32
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %arg7[%2] : memref<?xf32>
        %10 = arith.addi %arg9, %c1_i32 : i32
        scf.yield %10 : i32
      }
    }
    return
  }
}
