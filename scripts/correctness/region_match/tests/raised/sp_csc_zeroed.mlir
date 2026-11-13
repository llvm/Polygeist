module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @sp_csc_zeroed(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %0 {
      affine.store %cst, %arg6[%arg7] : memref<?xf64>
    }
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %1 {
      %2 = affine.load %arg2[%arg7] : memref<?xi32>
      %3 = scf.while (%arg8 = %2) : (i32) -> i32 {
        %4 = affine.load %arg2[%arg7 + 1] : memref<?xi32>
        %5 = arith.cmpi slt, %arg8, %4 : i32
        scf.condition(%5) %arg8 : i32
      } do {
      ^bb0(%arg8: i32):
        %4 = arith.index_cast %arg8 : i32 to index
        %5 = memref.load %arg3[%4] : memref<?xi32>
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg4[%4] : memref<?xf64>
        %8 = affine.load %arg5[%arg7] : memref<?xf64>
        %9 = arith.mulf %7, %8 : f64
        %10 = memref.load %arg6[%6] : memref<?xf64>
        %11 = arith.addf %10, %9 : f64
        memref.store %11, %arg6[%6] : memref<?xf64>
        %12 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %12 : i32
      }
    }
    return
  }
}
