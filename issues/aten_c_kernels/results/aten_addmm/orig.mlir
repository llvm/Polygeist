module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_addmm(%arg0: memref<?x16xf64>, %arg1: memref<?x16xf64>, %arg2: memref<?x16xf64>, %arg3: f64, %arg4: f64) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg5 = 0 to 16 {
      affine.for %arg6 = 0 to 16 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<?x16xf64>
        %1 = arith.mulf %0, %arg3 : f64
        affine.store %1, %arg2[%arg5, %arg6] : memref<?x16xf64>
      }
    }
    affine.for %arg5 = 0 to 16 {
      affine.for %arg6 = 0 to 16 {
        affine.for %arg7 = 0 to 16 {
          %0 = affine.load %arg0[%arg5, %arg7] : memref<?x16xf64>
          %1 = arith.mulf %arg4, %0 : f64
          %2 = affine.load %arg1[%arg7, %arg6] : memref<?x16xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg2[%arg5, %arg6] : memref<?x16xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg2[%arg5, %arg6] : memref<?x16xf64>
        }
      }
    }
    return
  }
}
