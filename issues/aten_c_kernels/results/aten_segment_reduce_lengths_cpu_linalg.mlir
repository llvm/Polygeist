#map = affine_map<()[s0] -> (s0 - 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_segment_reduce_lengths_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.cmpi eq, %arg2, %c2_i32 : i32
    %2 = arith.cmpi eq, %arg2, %c0_i32 : i32
    %3 = arith.cmpi eq, %arg2, %c3_i32 : i32
    %4 = arith.select %3, %cst_0, %cst_1 : f32
    %5 = arith.select %1, %cst, %4 : f32
    %6 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %7 = arith.select %2, %true, %6 : i1
    %8 = arith.addi %0, %c-1 : index
    %9 = arith.cmpi eq, %8, %c0 : index
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg4 = 0 to 16 {
      %10 = affine.load %alloca[] : memref<i32>
      %11 = affine.load %arg1[%arg4] : memref<?xi32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = arith.index_cast %10 : i32 to index
      %14 = arith.addi %13, %12 : index
      %15 = arith.index_cast %14 : index to i32
      %16 = scf.for %arg5 = %c0 to %12 step %c1 iter_args(%arg6 = %5) -> (f32) {
        %22 = arith.addi %13, %arg5 : index
        %23 = memref.load %arg0[%22] : memref<?xf32>
        %24 = scf.if %7 -> (f32) {
          %25 = arith.addf %arg6, %23 : f32
          scf.yield %25 : f32
        } else {
          %25 = affine.apply #map()[%0]
          %26 = arith.cmpi eq, %25, %c0 : index
          %27 = arith.cmpf ogt, %arg6, %23 : f32
          %28 = arith.select %27, %arg6, %23 : f32
          %29 = arith.cmpf olt, %arg6, %23 : f32
          %30 = arith.select %29, %arg6, %23 : f32
          %31 = arith.select %26, %28, %30 : f32
          scf.yield %31 : f32
        }
        scf.yield %24 : f32
      }
      %17 = arith.cmpi ne, %11, %c0_i32 : i32
      %18 = arith.andi %9, %17 : i1
      %19 = arith.sitofp %11 : i32 to f32
      %20 = arith.divf %16, %19 : f32
      %21 = arith.select %18, %20, %16 : f32
      affine.store %21, %arg3[%arg4] : memref<?xf32>
      affine.store %15, %alloca[] : memref<i32>
    }
    return
  }
}

