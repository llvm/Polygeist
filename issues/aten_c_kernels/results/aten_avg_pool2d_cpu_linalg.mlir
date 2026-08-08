#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 2 + 2)>
#map3 = affine_map<(d0)[s0, s1] -> (d0 + s0 * 7 + s1 * 42)>
#map4 = affine_map<(d0) -> ()>
#map5 = affine_map<(d0)[s0, s1] -> (d0 + s0 * 9 + s1 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 3 {
        %alloca = memref.alloca(%c3) : memref<?xi32>
        %alloca_0 = memref.alloca(%c3) : memref<?xf32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
        ^bb0(%out: i32):
          linalg.yield %c0_i32 : i32
        }
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca_0 : memref<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        }
        affine.for %arg4 = 0 to 3 {
          %1 = arith.index_cast %arg4 : index to i32
          %2 = arith.muli %1, %c2_i32 : i32
          %3 = arith.addi %2, %c2_i32 : i32
          %4 = arith.index_cast %3 : i32 to index
          %5 = arith.index_cast %2 : i32 to index
          %6 = arith.subi %4, %5 : index
          affine.for %arg5 = #map1(%arg3) to #map2(%arg3) {
            %7 = affine.load %alloca[%arg4] : memref<?xi32>
            %8 = arith.index_cast %7 : i32 to index
            %9 = arith.addi %8, %6 : index
            %10 = arith.index_cast %9 : index to i32
            %11 = polygeist.submap(%arg0, %arg5, %arg2, %c6) {map = #map3} : (memref<?xf32>, index, index, index) -> memref<?xf32>
            %subview = memref.subview %alloca_0[%arg4] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["reduction"]} ins(%11 : memref<?xf32>) outs(%subview : memref<f32, strided<[], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %12 = arith.addf %out, %in : f32
              %13 = linalg.index 0 : index
              %14 = affine.apply #map1(%arg4)
              %15 = arith.cmpi sge, %13, %14 : index
              %16 = affine.apply #map2(%arg4)
              %17 = arith.cmpi slt, %13, %16 : index
              %18 = arith.andi %15, %17 : i1
              %19 = arith.select %18, %12, %out : f32
              linalg.yield %19 : f32
            }
            affine.store %10, %alloca[%arg4] : memref<?xi32>
          }
        } {polygeist.was_parallel}
        %0 = polygeist.submap(%arg1, %arg2, %arg3, %c3) {map = #map5} : (memref<?xf32>, index, index, index) -> memref<?xf32>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloca, %alloca_0 : memref<?xi32>, memref<?xf32>) outs(%0 : memref<?xf32>) {
        ^bb0(%in: i32, %in_1: f32, %out: f32):
          %1 = arith.sitofp %in : i32 to f32
          %2 = arith.divf %in_1, %1 : f32
          linalg.yield %2 : f32
        }
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

