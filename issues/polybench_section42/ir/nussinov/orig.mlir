#map = affine_map<(d0)[s0] -> (-d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set2 = affine_set<(d0, d1)[s0] : (d0 + d1 - s0 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_nussinov(%arg0: i32, %arg1: memref<?xi8>, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = #map(%arg3)[%0] to %0 {
        affine.if #set(%arg4) {
          %1 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %2 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4 - 1] : memref<?x?xf64>
          %3 = arith.cmpf oge, %1, %2 : f64
          %4 = scf.if %3 -> (f64) {
            %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            scf.yield %5 : f64
          } else {
            %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4 - 1] : memref<?x?xf64>
            scf.yield %5 : f64
          }
          affine.store %4, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
        affine.if #set(%arg3) {
          %1 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %2 = affine.load %arg2[-%arg3 + symbol(%0), %arg4] : memref<?x?xf64>
          %3 = arith.cmpf oge, %1, %2 : f64
          %4 = scf.if %3 -> (f64) {
            %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            scf.yield %5 : f64
          } else {
            %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4] : memref<?x?xf64>
            scf.yield %5 : f64
          }
          affine.store %4, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
        affine.if #set1(%arg4, %arg3) {
          affine.if #set2(%arg3, %arg4)[%0] {
            %1 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            %2 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
            %3 = affine.load %arg1[-%arg3 + symbol(%0) - 1] : memref<?xi8>
            %4 = arith.extsi %3 : i8 to i32
            %5 = affine.load %arg1[%arg4] : memref<?xi8>
            %6 = arith.extsi %5 : i8 to i32
            %7 = arith.addi %4, %6 : i32
            %8 = arith.cmpi eq, %7, %c3_i32 : i32
            %9 = arith.extui %8 : i1 to i32
            %10 = arith.sitofp %9 : i32 to f64
            %11 = arith.addf %2, %10 : f64
            %12 = arith.cmpf oge, %1, %11 : f64
            %13 = scf.if %12 -> (f64) {
              %14 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
              scf.yield %14 : f64
            } else {
              %14 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
              %15 = affine.load %arg1[-%arg3 + symbol(%0) - 1] : memref<?xi8>
              %16 = arith.extsi %15 : i8 to i32
              %17 = arith.addi %16, %6 : i32
              %18 = arith.cmpi eq, %17, %c3_i32 : i32
              %19 = arith.extui %18 : i1 to i32
              %20 = arith.sitofp %19 : i32 to f64
              %21 = arith.addf %14, %20 : f64
              scf.yield %21 : f64
            }
            affine.store %13, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          } else {
            %1 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            %2 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
            %3 = arith.cmpf oge, %1, %2 : f64
            %4 = scf.if %3 -> (f64) {
              %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
              scf.yield %5 : f64
            } else {
              %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
              scf.yield %5 : f64
            }
            affine.store %4, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          }
        }
        affine.for %arg5 = #map(%arg3)[%0] to #map1(%arg4) {
          %1 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %2 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg5] : memref<?x?xf64>
          %3 = affine.load %arg2[%arg5 + 1, %arg4] : memref<?x?xf64>
          %4 = arith.addf %2, %3 : f64
          %5 = arith.cmpf oge, %1, %4 : f64
          %6 = scf.if %5 -> (f64) {
            %7 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            scf.yield %7 : f64
          } else {
            %7 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg5] : memref<?x?xf64>
            %8 = arith.addf %7, %3 : f64
            scf.yield %8 : f64
          }
          affine.store %6, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
