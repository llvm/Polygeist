#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1)>
#map2 = affine_map<(d0, d1) -> (d1 + 1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d1 + 2)>
#map5 = affine_map<(d0, d1) -> (d0 + 2)>
#map6 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.000000e-01 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = affine.apply #map()[%0]
    %3 = affine.apply #map()[%0]
    %4 = arith.subi %3, %c1 : index
    %5 = affine.apply #map()[%0]
    %6 = arith.subi %5, %c1 : index
    %7 = arith.subi %2, %c1 : index
    %8 = affine.apply #map()[%0]
    %9 = arith.subi %8, %c1 : index
    affine.for %arg4 = 0 to %1 {
      scf.for %arg5 = %c0 to %6 step %c1 {
        scf.for %arg6 = %c0 to %4 step %c1 {
          %10 = affine.apply #map1(%arg5, %arg6)
          %11 = affine.apply #map2(%arg5, %arg6)
          %12 = memref.load %arg2[%10, %11] : memref<?x?xf64>
          %13 = affine.apply #map1(%arg5, %arg6)
          %14 = affine.apply #map3(%arg5, %arg6)
          %15 = memref.load %arg2[%13, %14] : memref<?x?xf64>
          %16 = affine.apply #map1(%arg5, %arg6)
          %17 = affine.apply #map4(%arg5, %arg6)
          %18 = memref.load %arg2[%16, %17] : memref<?x?xf64>
          %19 = affine.apply #map5(%arg5, %arg6)
          %20 = affine.apply #map2(%arg5, %arg6)
          %21 = memref.load %arg2[%19, %20] : memref<?x?xf64>
          %22 = affine.apply #map6(%arg5, %arg6)
          %23 = affine.apply #map2(%arg5, %arg6)
          %24 = memref.load %arg2[%22, %23] : memref<?x?xf64>
          %25 = arith.addf %12, %15 : f64
          %26 = arith.addf %25, %18 : f64
          %27 = arith.addf %26, %21 : f64
          %28 = arith.addf %27, %24 : f64
          %29 = arith.mulf %28, %cst : f64
          %30 = affine.apply #map1(%arg5, %arg6)
          %31 = affine.apply #map2(%arg5, %arg6)
          memref.store %29, %arg3[%30, %31] : memref<?x?xf64>
        }
      }
      scf.for %arg5 = %c0 to %9 step %c1 {
        scf.for %arg6 = %c0 to %7 step %c1 {
          %10 = affine.apply #map1(%arg5, %arg6)
          %11 = affine.apply #map2(%arg5, %arg6)
          %12 = memref.load %arg3[%10, %11] : memref<?x?xf64>
          %13 = affine.apply #map1(%arg5, %arg6)
          %14 = affine.apply #map3(%arg5, %arg6)
          %15 = memref.load %arg3[%13, %14] : memref<?x?xf64>
          %16 = affine.apply #map1(%arg5, %arg6)
          %17 = affine.apply #map4(%arg5, %arg6)
          %18 = memref.load %arg3[%16, %17] : memref<?x?xf64>
          %19 = affine.apply #map5(%arg5, %arg6)
          %20 = affine.apply #map2(%arg5, %arg6)
          %21 = memref.load %arg3[%19, %20] : memref<?x?xf64>
          %22 = affine.apply #map6(%arg5, %arg6)
          %23 = affine.apply #map2(%arg5, %arg6)
          %24 = memref.load %arg3[%22, %23] : memref<?x?xf64>
          %25 = arith.addf %12, %15 : f64
          %26 = arith.addf %25, %18 : f64
          %27 = arith.addf %26, %21 : f64
          %28 = arith.addf %27, %24 : f64
          %29 = arith.mulf %28, %cst : f64
          %30 = affine.apply #map1(%arg5, %arg6)
          %31 = affine.apply #map2(%arg5, %arg6)
          memref.store %29, %arg2[%30, %31] : memref<?x?xf64>
        }
      }
    }
    return
  }
}

