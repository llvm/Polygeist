#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 6)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 * 2 + 2)>
#map6 = affine_map<(d0)[s0] -> (d0 + s0 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_max_pool1d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    affine.for %arg3 = 0 to 2 {
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
      %0 = polygeist.submap(%arg0, %arg3, %c3, %c6) {map = #map1} : (memref<?xf32>, index, index, index) -> memref<?x?xf32>
      %subview = memref.subview %alloca[0] [%c3] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
      %subview_1 = memref.subview %alloca_0[0] [%c3] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%0 : memref<?x?xf32>) outs(%subview, %subview_1 : memref<?xi32, strided<[1]>>, memref<?xf32, strided<[1]>>) {
      ^bb0(%in: f32, %out: i32, %out_2: f32):
        %3 = linalg.index 0 : index
        %4 = linalg.index 1 : index
        %5 = arith.index_cast %4 : index to i32
        %6 = arith.cmpf ogt, %in, %out_2 : f32
        %7 = arith.select %6, %5, %out : i32
        %8 = arith.select %6, %in, %out_2 : f32
        %9 = linalg.index 1 : index
        %10 = affine.apply #map4(%3)
        %11 = arith.cmpi sge, %9, %10 : index
        %12 = affine.apply #map5(%3)
        %13 = arith.cmpi slt, %9, %12 : index
        %14 = arith.andi %11, %13 : i1
        %15 = arith.select %14, %7, %out : i32
        %16 = arith.select %14, %8, %out_2 : f32
        linalg.yield %15, %16 : i32, f32
      }
      %1 = polygeist.submap(%arg1, %arg3, %c3) {map = #map6} : (memref<?xf32>, index, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca_0 : memref<?xf32>) outs(%1 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      %2 = polygeist.submap(%arg2, %arg3, %c3) {map = #map6} : (memref<?xi32>, index, index) -> memref<?xi32>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloca : memref<?xi32>) outs(%2 : memref<?xi32>) {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      }
    } {polygeist.was_parallel}
    return
  }
}

