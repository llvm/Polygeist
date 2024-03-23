module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z6gatherPdS_Piii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.sitofp %arg4 : i32 to f64
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg2[%arg5] : memref<?xi32>
      %3 = arith.muli %2, %arg4 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %arg1[%4] : memref<?xf64>
      %6 = arith.mulf %1, %5 : f64
      %7 = affine.load %arg0[%arg5] : memref<?xf64>
      %8 = arith.addf %7, %6 : f64
      affine.store %8, %arg0[%arg5] : memref<?xf64>
    }
    return
  }
  func.func @_Z7scatterPdS_PiS0_ii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.sitofp %arg5 : i32 to f64
    affine.for %arg6 = 0 to %0 {
      %2 = affine.load %arg3[%arg6] : memref<?xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = memref.load %arg2[%3] : memref<?xi32>
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.load %arg1[%arg6] : memref<?xf64>
      %7 = arith.mulf %1, %6 : f64
      %8 = memref.load %arg0[%5] : memref<?xf64>
      %9 = arith.addf %8, %7 : f64
      memref.store %9, %arg0[%5] : memref<?xf64>
    }
    return
  }
}

