module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cross_cpu_backend(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 256 {
      %0 = affine.load %arg0[%arg3, 1] : memref<?x3xf32>
      %1 = affine.load %arg1[%arg3, 2] : memref<?x3xf32>
      %2 = arith.mulf %0, %1 : f32
      %3 = affine.load %arg0[%arg3, 2] : memref<?x3xf32>
      %4 = affine.load %arg1[%arg3, 1] : memref<?x3xf32>
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.subf %2, %5 : f32
      affine.store %6, %arg2[%arg3, 0] : memref<?x3xf32>
      %7 = affine.load %arg0[%arg3, 2] : memref<?x3xf32>
      %8 = affine.load %arg1[%arg3, 0] : memref<?x3xf32>
      %9 = arith.mulf %7, %8 : f32
      %10 = affine.load %arg0[%arg3, 0] : memref<?x3xf32>
      %11 = affine.load %arg1[%arg3, 2] : memref<?x3xf32>
      %12 = arith.mulf %10, %11 : f32
      %13 = arith.subf %9, %12 : f32
      affine.store %13, %arg2[%arg3, 1] : memref<?x3xf32>
      %14 = affine.load %arg0[%arg3, 0] : memref<?x3xf32>
      %15 = affine.load %arg1[%arg3, 1] : memref<?x3xf32>
      %16 = arith.mulf %14, %15 : f32
      %17 = affine.load %arg0[%arg3, 1] : memref<?x3xf32>
      %18 = affine.load %arg1[%arg3, 0] : memref<?x3xf32>
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.subf %16, %19 : f32
      affine.store %20, %arg2[%arg3, 2] : memref<?x3xf32>
    }
    return
  }
}
