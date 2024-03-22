module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : memref<100xf32> = uninitialized
  func.func @_Z14uselessinstr1sv() attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.887000e+03 : f32
    %0 = memref.get_global @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : memref<100xf32>
    affine.store %cst, %0[5] : memref<100xf32>
    return
  }
  func.func @_Z6gatherPdS_Piii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.887000e+03 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.sitofp %arg4 : i32 to f64
    %3 = memref.get_global @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : memref<100xf32>
    affine.for %arg5 = 0 to %1 {
      affine.store %cst, %3[5] : memref<100xf32>
      %4 = affine.load %arg2[%arg5 + symbol(%0)] : memref<?xi32>
      %5 = arith.muli %4, %arg4 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7 = memref.load %arg1[%6] : memref<?xf64>
      %8 = arith.mulf %2, %7 : f64
      %9 = affine.load %arg0[%arg5] : memref<?xf64>
      %10 = arith.addf %9, %8 : f64
      affine.store %10, %arg0[%arg5] : memref<?xf64>
      affine.store %cst, %3[5] : memref<100xf32>
    }
    return
  }
  func.func @_Z7scatterPdS_Piii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.sitofp %arg4 : i32 to f64
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg2[%arg5] : memref<?xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = affine.load %arg1[%arg5] : memref<?xf64>
      %5 = arith.mulf %1, %4 : f64
      %6 = memref.load %arg0[%3] : memref<?xf64>
      %7 = arith.addf %6, %5 : f64
      memref.store %7, %arg0[%3] : memref<?xf64>
    }
    return
  }
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.887000e+03 : f32
    %c2_i32 = arith.constant 2 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<10xi32>
    %alloca_1 = memref.alloca() : memref<10xf64>
    %alloca_2 = memref.alloca() : memref<10xf64>
    %1 = memref.get_global @"_Z14uselessinstr1sv@static@_ZZ14uselessinstr1svE1d" : memref<100xf32>
    affine.store %cst_0, %1[5] : memref<100xf32>
    affine.store %cst_0, %1[5] : memref<100xf32>
    affine.for %arg0 = 0 to 10 {
      %2 = affine.load %alloca[%arg0 + 2] : memref<10xi32>
      %3 = arith.muli %2, %c2_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %alloca_1[%4] : memref<10xf64>
      %6 = arith.mulf %5, %cst : f64
      %7 = affine.load %alloca_2[%arg0] : memref<10xf64>
      %8 = arith.addf %7, %6 : f64
      affine.store %8, %alloca_2[%arg0] : memref<10xf64>
    }
    affine.for %arg0 = 0 to 10 {
      %2 = affine.load %alloca[%arg0] : memref<10xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = affine.load %alloca_1[%arg0] : memref<10xf64>
      %5 = arith.mulf %4, %cst : f64
      %6 = memref.load %alloca_2[%3] : memref<10xf64>
      %7 = arith.addf %6, %5 : f64
      memref.store %7, %alloca_2[%3] : memref<10xf64>
    }
    return %0 : i32
  }
}

