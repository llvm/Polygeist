#map = affine_map<(d0)[s0] -> (-d0 + s0)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map2 = affine_map<(d0)[s0, s1, s2] -> (-s0 + s2 - 1, s1)>
#map3 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set2 = affine_set<(d0, d1)[s0] : (d0 + d1 - s0 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_nussinov(%arg0: i32, %arg1: memref<?xi8>, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.subi %0, %c1 : index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = #map(%arg3)[%0] to %0 {
        affine.if #set(%arg4) {
          %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4 - 1] : memref<?x?xf64>
          %6 = arith.cmpf oge, %4, %5 : f64
          %7 = arith.select %6, %4, %5 : f64
          affine.store %7, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
        affine.if #set(%arg3) {
          %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4] : memref<?x?xf64>
          %6 = arith.cmpf oge, %4, %5 : f64
          %7 = arith.select %6, %4, %5 : f64
          affine.store %7, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
        affine.if #set1(%arg4, %arg3) {
          affine.if #set2(%arg3, %arg4)[%0] {
            %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
            %6 = affine.load %arg1[-%arg3 + symbol(%0) - 1] : memref<?xi8>
            %7 = arith.extsi %6 : i8 to i32
            %8 = affine.load %arg1[%arg4] : memref<?xi8>
            %9 = arith.extsi %8 : i8 to i32
            %10 = arith.addi %7, %9 : i32
            %11 = arith.cmpi eq, %10, %c3_i32 : i32
            %12 = arith.extui %11 : i1 to i32
            %13 = arith.sitofp %12 : i32 to f64
            %14 = arith.addf %5, %13 : f64
            %15 = arith.cmpf oge, %4, %14 : f64
            %16 = arith.extsi %6 : i8 to i32
            %17 = arith.addi %16, %9 : i32
            %18 = arith.cmpi eq, %17, %c3_i32 : i32
            %19 = arith.extui %18 : i1 to i32
            %20 = arith.sitofp %19 : i32 to f64
            %21 = arith.addf %5, %20 : f64
            %22 = arith.select %15, %4, %21 : f64
            affine.store %22, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          } else {
            %4 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            %5 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
            %6 = arith.cmpf oge, %4, %5 : f64
            %7 = arith.select %6, %4, %5 : f64
            affine.store %7, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          }
        }
        %2 = polygeist.submap(%arg2, %arg3, %0, %1) {map = #map1} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %subview = memref.subview %arg2[1, %arg4] [%1, 1] [1, 1] : memref<?x?xf64> to memref<?xf64, strided<[?], offset: ?>>
        %3 = polygeist.submap(%arg2, %arg3, %arg4, %0, %1) {map = #map2} : (memref<?x?xf64>, index, index, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["reduction"]} ins(%2, %subview : memref<?xf64>, memref<?xf64, strided<[?], offset: ?>>) outs(%3 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %4 = arith.addf %in, %in_0 : f64
          %5 = arith.cmpf oge, %out, %4 : f64
          %6 = arith.addf %in, %in_0 : f64
          %7 = arith.select %5, %out, %6 : f64
          %8 = linalg.index 0 : index
          %9 = affine.apply #map(%arg3)[%0]
          %10 = arith.cmpi sge, %8, %9 : index
          %11 = arith.cmpi slt, %8, %arg4 : index
          %12 = arith.andi %10, %11 : i1
          %13 = arith.select %12, %7, %out : f64
          linalg.yield %13 : f64
        }
      }
    }
    return
  }
}

