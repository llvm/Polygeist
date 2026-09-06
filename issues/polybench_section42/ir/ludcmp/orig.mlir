#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_ludcmp(%arg0: i32, %arg1: memref<?x?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    %alloca = memref.alloca() : memref<f64>
    %1 = llvm.mlir.undef : f64
    affine.store %1, %alloca[] : memref<f64>
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to #map(%arg5) {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<?x?xf64>
        affine.store %2, %alloca[] : memref<f64>
        affine.for %arg7 = 0 to #map(%arg6) {
          %6 = affine.load %arg1[%arg5, %arg7] : memref<?x?xf64>
          %7 = affine.load %arg1[%arg7, %arg6] : memref<?x?xf64>
          %8 = arith.mulf %6, %7 : f64
          %9 = affine.load %alloca[] : memref<f64>
          %10 = arith.subf %9, %8 : f64
          affine.store %10, %alloca[] : memref<f64>
        }
        %3 = affine.load %alloca[] : memref<f64>
        %4 = affine.load %arg1[%arg6, %arg6] : memref<?x?xf64>
        %5 = arith.divf %3, %4 : f64
        affine.store %5, %arg1[%arg5, %arg6] : memref<?x?xf64>
      }
      affine.for %arg6 = #map(%arg5) to %0 {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<?x?xf64>
        affine.store %2, %alloca[] : memref<f64>
        affine.for %arg7 = 0 to #map(%arg5) {
          %4 = affine.load %arg1[%arg5, %arg7] : memref<?x?xf64>
          %5 = affine.load %arg1[%arg7, %arg6] : memref<?x?xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %alloca[] : memref<f64>
          %8 = arith.subf %7, %6 : f64
          affine.store %8, %alloca[] : memref<f64>
        }
        %3 = affine.load %alloca[] : memref<f64>
        affine.store %3, %arg1[%arg5, %arg6] : memref<?x?xf64>
      }
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg2[%arg5] : memref<?xf64>
      affine.store %2, %alloca[] : memref<f64>
      affine.for %arg6 = 0 to #map(%arg5) {
        %4 = affine.load %arg1[%arg5, %arg6] : memref<?x?xf64>
        %5 = affine.load %arg4[%arg6] : memref<?xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = affine.load %alloca[] : memref<f64>
        %8 = arith.subf %7, %6 : f64
        affine.store %8, %alloca[] : memref<f64>
      }
      %3 = affine.load %alloca[] : memref<f64>
      affine.store %3, %arg4[%arg5] : memref<?xf64>
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg4[-%arg5 + symbol(%0) - 1] : memref<?xf64>
      affine.store %2, %alloca[] : memref<f64>
      affine.for %arg6 = #map1(%arg5)[%0] to %0 {
        %6 = affine.load %arg1[-%arg5 + symbol(%0) - 1, %arg6] : memref<?x?xf64>
        %7 = affine.load %arg3[%arg6] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = affine.load %alloca[] : memref<f64>
        %10 = arith.subf %9, %8 : f64
        affine.store %10, %alloca[] : memref<f64>
      }
      %3 = affine.load %alloca[] : memref<f64>
      %4 = affine.load %arg1[-%arg5 + symbol(%0) - 1, -%arg5 + symbol(%0) - 1] : memref<?x?xf64>
      %5 = arith.divf %3, %4 : f64
      affine.store %5, %arg3[-%arg5 + symbol(%0) - 1] : memref<?xf64>
    }
    return
  }
}
