module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg3[%arg12, %arg13] : memref<?x?xf64>
        %2 = affine.load %arg4[%arg12] : memref<?xf64>
        %3 = affine.load %arg5[%arg13] : memref<?xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %1, %4 : f64
        %6 = affine.load %arg6[%arg12] : memref<?xf64>
        %7 = affine.load %arg7[%arg13] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %5, %8 : f64
        affine.store %9, %arg3[%arg12, %arg13] : memref<?x?xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg9[%arg12] : memref<?xf64>
        %2 = affine.load %arg3[%arg13, %arg12] : memref<?x?xf64>
        %3 = arith.mulf %arg2, %2 : f64
        %4 = affine.load %arg10[%arg13] : memref<?xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = arith.addf %1, %5 : f64
        affine.store %6, %arg9[%arg12] : memref<?xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<?xf64>
      %2 = affine.load %arg11[%arg12] : memref<?xf64>
      %3 = arith.addf %1, %2 : f64
      affine.store %3, %arg9[%arg12] : memref<?xf64>
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg8[%arg12] : memref<?xf64>
        %2 = affine.load %arg3[%arg12, %arg13] : memref<?x?xf64>
        %3 = arith.mulf %arg1, %2 : f64
        %4 = affine.load %arg9[%arg13] : memref<?xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = arith.addf %1, %5 : f64
        affine.store %6, %arg8[%arg12] : memref<?xf64>
      }
    }
    return
  }
}
