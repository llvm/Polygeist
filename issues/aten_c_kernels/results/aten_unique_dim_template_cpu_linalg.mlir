#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unique_dim_template_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c127 = arith.constant 127 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 128 {
      affine.store %c1_i32, %arg1[%arg2] : memref<?xi32>
      %alloca = memref.alloca(%arg2) : memref<?xi32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloca : memref<?xi32>) {
      ^bb0(%out: i32):
        %0 = linalg.index 0 : index
        %1 = arith.cmpi slt, %0, %arg2 : index
        %2 = arith.select %1, %c1_i32, %out : i32
        linalg.yield %2 : i32
      }
      %subview = memref.subview %arg0[%arg2, 0] [1, %c16] [1, 1] : memref<?x16xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %arg0[0, 0] [%c127, %c16] [1, 1] : memref<?x16xf32> to memref<?x?xf32, strided<[16, 1]>>
      %subview_1 = memref.subview %alloca[0] [%c127] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
      linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>>, memref<?x?xf32, strided<[16, 1]>>) outs(%subview_1 : memref<?xi32, strided<[1]>>) {
      ^bb0(%in: f32, %in_4: f32, %out: i32):
        %0 = arith.cmpf oeq, %in, %in_4 : f32
        %1 = arith.extui %0 : i1 to i32
        %2 = arith.andi %out, %1 : i32
        linalg.yield %2 : i32
      }
      %subview_2 = memref.subview %alloca[0] [%c127] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
      %subview_3 = memref.subview %arg1[%arg2] [1] [1] : memref<?xi32> to memref<i32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["reduction"]} ins(%subview_2 : memref<?xi32, strided<[1]>>) outs(%subview_3 : memref<i32, strided<[], offset: ?>>) {
      ^bb0(%in: i32, %out: i32):
        %0 = arith.cmpi eq, %in, %c0_i32 : i32
        %1 = arith.extui %0 : i1 to i32
        %2 = arith.andi %out, %1 : i32
        %3 = linalg.index 0 : index
        %4 = arith.cmpi slt, %3, %arg2 : index
        %5 = arith.select %4, %2, %out : i32
        linalg.yield %5 : i32
      }
    } {polygeist.was_parallel}
    return
  }
}

