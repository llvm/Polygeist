#map = affine_map<(d0)[s0] -> (-d0 + s0)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 1, s0)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (-s0 + s2 - 1, s1)>
#map4 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 - 1 >= 0, d1 - 1 >= 0)>
#set2 = affine_set<(d0, d1)[s0] : (d0 + d1 - s0 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_nussinov(%arg0: i32, %arg1: memref<?xi8>, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.subi %0, %c1 : index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = #map(%arg3)[%0] to %0 {
        affine.if #set(%arg4) {
          %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %6 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4 - 1] : memref<?x?xf64>
          %7 = arith.cmpf oge, %5, %6 : f64
          %8 = arith.select %7, %5, %6 : f64
          affine.store %8, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
        affine.if #set(%arg3) {
          %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          %6 = affine.load %arg2[-%arg3 + symbol(%0), %arg4] : memref<?x?xf64>
          %7 = arith.cmpf oge, %5, %6 : f64
          %8 = arith.select %7, %5, %6 : f64
          affine.store %8, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
        }
        affine.if #set1(%arg4, %arg3) {
          affine.if #set2(%arg3, %arg4)[%0] {
            %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            %6 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
            %7 = affine.load %arg1[-%arg3 + symbol(%0) - 1] : memref<?xi8>
            %8 = arith.extsi %7 : i8 to i32
            %9 = affine.load %arg1[%arg4] : memref<?xi8>
            %10 = arith.extsi %9 : i8 to i32
            %11 = arith.addi %8, %10 : i32
            %12 = arith.cmpi eq, %11, %c3_i32 : i32
            %13 = arith.extui %12 : i1 to i32
            %14 = arith.sitofp %13 : i32 to f64
            %15 = arith.addf %6, %14 : f64
            %16 = arith.cmpf oge, %5, %15 : f64
            %17 = arith.extsi %7 : i8 to i32
            %18 = arith.addi %17, %10 : i32
            %19 = arith.cmpi eq, %18, %c3_i32 : i32
            %20 = arith.extui %19 : i1 to i32
            %21 = arith.sitofp %20 : i32 to f64
            %22 = arith.addf %6, %21 : f64
            %23 = arith.select %16, %5, %22 : f64
            affine.store %23, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          } else {
            %5 = affine.load %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
            %6 = affine.load %arg2[-%arg3 + symbol(%0), %arg4 - 1] : memref<?x?xf64>
            %7 = arith.cmpf oge, %5, %6 : f64
            %8 = arith.select %7, %5, %6 : f64
            affine.store %8, %arg2[-%arg3 + symbol(%0) - 1, %arg4] : memref<?x?xf64>
          }
        }
        %2 = polygeist.submap(%arg2, %arg3, %0, %1) {map = #map1} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %3 = polygeist.submap(%arg2, %arg4, %1) {map = #map2} : (memref<?x?xf64>, index, index) -> memref<?xf64>
        %4 = polygeist.submap(%arg2, %arg3, %arg4, %0, %1) {map = #map3} : (memref<?x?xf64>, index, index, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["reduction"]} ins(%2, %3 : memref<?xf64>, memref<?xf64>) outs(%4 : memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %5 = arith.addf %in, %in_0 : f64
          %6 = arith.cmpf oge, %out, %5 : f64
          %7 = arith.addf %in, %in_0 : f64
          %8 = arith.select %6, %out, %7 : f64
          %9 = linalg.index 0 : index
          %10 = affine.apply #map(%arg3)[%0]
          %11 = arith.cmpi sge, %9, %10 : index
          %12 = arith.cmpi slt, %9, %arg4 : index
          %13 = arith.andi %11, %12 : i1
          %14 = arith.select %13, %8, %out : f64
          linalg.yield %14 : f64
        }
      }
    }
    return
  }
}

