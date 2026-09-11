module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x8x12xf64>, %arg4: memref<?x12xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = 0 to %1 {
          affine.store %cst, %arg5[%arg8] : memref<?xf64>
          affine.for %arg9 = 0 to %1 {
            %3 = affine.load %arg3[%arg6, %arg7, %arg9] : memref<?x8x12xf64>
            %4 = affine.load %arg4[%arg9, %arg8] : memref<?x12xf64>
            %5 = arith.mulf %3, %4 : f64
            %6 = affine.load %arg5[%arg8] : memref<?xf64>
            %7 = arith.addf %6, %5 : f64
            affine.store %7, %arg5[%arg8] : memref<?xf64>
          }
        }
        affine.for %arg8 = 0 to %1 {
          %3 = affine.load %arg5[%arg8] : memref<?xf64>
          affine.store %3, %arg3[%arg6, %arg7, %arg8] : memref<?x8x12xf64>
        }
      }
    }
    return
  }
}
