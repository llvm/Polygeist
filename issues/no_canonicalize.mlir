#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e-01 : f64
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<1300x1300xf32>
    %alloca_0 = memref.alloca() : memref<1300x1300xf32>
    affine.for %arg0 = 0 to 500 {
      affine.for %arg1 = 1 to 1299 {
        affine.for %arg2 = 1 to 1299 {
          %0 = affine.load %alloca_0[%arg1, %arg2] : memref<1300x1300xf32>
          %1 = affine.load %alloca_0[%arg1, %arg2 - 1] : memref<1300x1300xf32>
          %2 = arith.addf %0, %1 : f32
          %3 = affine.load %alloca_0[%arg1, %arg2 + 1] : memref<1300x1300xf32>
          %4 = arith.addf %2, %3 : f32
          %5 = affine.load %alloca_0[%arg1 + 1, %arg2] : memref<1300x1300xf32>
          %6 = arith.addf %4, %5 : f32
          %7 = affine.load %alloca_0[%arg1 - 1, %arg2] : memref<1300x1300xf32>
          %8 = arith.addf %6, %7 : f32
          %9 = arith.extf %8 : f32 to f64
          %10 = arith.mulf %9, %cst : f64
          %11 = arith.truncf %10 : f64 to f32
          affine.store %11, %alloca[%arg1, %arg2] : memref<1300x1300xf32>
        }
      }
    }
    return %c0_i32 : i32
  }
}
