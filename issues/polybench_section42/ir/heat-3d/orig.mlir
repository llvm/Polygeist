#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<?x?x?xf64>, %arg3: memref<?x?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.250000e-01 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to 501 {
      affine.for %arg5 = 1 to #map()[%0] {
        affine.for %arg6 = 1 to #map()[%0] {
          affine.for %arg7 = 1 to #map()[%0] {
            %1 = affine.load %arg2[%arg5 + 1, %arg6, %arg7] : memref<?x?x?xf64>
            %2 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
            %3 = arith.mulf %2, %cst_0 : f64
            %4 = arith.subf %1, %3 : f64
            %5 = affine.load %arg2[%arg5 - 1, %arg6, %arg7] : memref<?x?x?xf64>
            %6 = arith.addf %4, %5 : f64
            %7 = arith.mulf %6, %cst : f64
            %8 = affine.load %arg2[%arg5, %arg6 + 1, %arg7] : memref<?x?x?xf64>
            %9 = arith.subf %8, %3 : f64
            %10 = affine.load %arg2[%arg5, %arg6 - 1, %arg7] : memref<?x?x?xf64>
            %11 = arith.addf %9, %10 : f64
            %12 = arith.mulf %11, %cst : f64
            %13 = arith.addf %7, %12 : f64
            %14 = affine.load %arg2[%arg5, %arg6, %arg7 + 1] : memref<?x?x?xf64>
            %15 = arith.subf %14, %3 : f64
            %16 = affine.load %arg2[%arg5, %arg6, %arg7 - 1] : memref<?x?x?xf64>
            %17 = arith.addf %15, %16 : f64
            %18 = arith.mulf %17, %cst : f64
            %19 = arith.addf %13, %18 : f64
            %20 = arith.addf %19, %2 : f64
            affine.store %20, %arg3[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
          }
        }
      }
      affine.for %arg5 = 1 to #map()[%0] {
        affine.for %arg6 = 1 to #map()[%0] {
          affine.for %arg7 = 1 to #map()[%0] {
            %1 = affine.load %arg3[%arg5 + 1, %arg6, %arg7] : memref<?x?x?xf64>
            %2 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
            %3 = arith.mulf %2, %cst_0 : f64
            %4 = arith.subf %1, %3 : f64
            %5 = affine.load %arg3[%arg5 - 1, %arg6, %arg7] : memref<?x?x?xf64>
            %6 = arith.addf %4, %5 : f64
            %7 = arith.mulf %6, %cst : f64
            %8 = affine.load %arg3[%arg5, %arg6 + 1, %arg7] : memref<?x?x?xf64>
            %9 = arith.subf %8, %3 : f64
            %10 = affine.load %arg3[%arg5, %arg6 - 1, %arg7] : memref<?x?x?xf64>
            %11 = arith.addf %9, %10 : f64
            %12 = arith.mulf %11, %cst : f64
            %13 = arith.addf %7, %12 : f64
            %14 = affine.load %arg3[%arg5, %arg6, %arg7 + 1] : memref<?x?x?xf64>
            %15 = arith.subf %14, %3 : f64
            %16 = affine.load %arg3[%arg5, %arg6, %arg7 - 1] : memref<?x?x?xf64>
            %17 = arith.addf %15, %16 : f64
            %18 = arith.mulf %17, %cst : f64
            %19 = arith.addf %13, %18 : f64
            %20 = arith.addf %19, %2 : f64
            affine.store %20, %arg2[%arg5, %arg6, %arg7] : memref<?x?x?xf64>
          }
        }
      }
    }
    return
  }
}
