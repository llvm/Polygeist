module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_aminmax_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<f32>
    %0 = llvm.mlir.undef : f32
    affine.store %0, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    affine.store %0, %alloca_0[] : memref<f32>
    %1 = affine.load %arg0[0] : memref<?xf32>
    affine.store %1, %alloca_0[] : memref<f32>
    %2 = affine.load %arg0[0] : memref<?xf32>
    affine.store %2, %alloca[] : memref<f32>
    affine.for %arg3 = 1 to 4096 {
      %5 = affine.load %arg0[%arg3] : memref<?xf32>
      %6 = affine.load %alloca_0[] : memref<f32>
      %7 = arith.cmpf olt, %5, %6 : f32
      %8 = arith.select %7, %5, %6 : f32
      affine.store %8, %alloca_0[] : memref<f32>
      %9 = affine.load %arg0[%arg3] : memref<?xf32>
      %10 = affine.load %alloca[] : memref<f32>
      %11 = arith.cmpf ogt, %9, %10 : f32
      %12 = arith.select %11, %9, %10 : f32
      affine.store %12, %alloca[] : memref<f32>
    }
    %3 = affine.load %alloca_0[] : memref<f32>
    affine.store %3, %arg1[0] : memref<?xf32>
    %4 = affine.load %alloca[] : memref<f32>
    affine.store %4, %arg2[0] : memref<?xf32>
    return
  }
}
