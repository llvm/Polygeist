#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_isin_default_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c257 = arith.constant 257 : index
    %c4096 = arith.constant 4096 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %alloca = memref.alloca(%c4096) : memref<?xi32>
    %1 = bufferization.to_tensor %alloca : memref<?xi32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%1 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %3 = bufferization.to_memref %2 : memref<?xi32>
    memref.copy %3, %alloca : memref<?xi32> to memref<?xi32>
    %subview = memref.subview %arg1[0] [%c257] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %4 = polygeist.submap(%arg0, %c4096) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %subview_0 = memref.subview %4[0] [%c4096] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    %subview_1 = memref.subview %alloca[0] [%c4096] [1] : memref<?xi32> to memref<?xi32, strided<[1]>>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "reduction"]} ins(%subview_0, %subview : memref<?xf32, strided<[1]>>, memref<?xf32, strided<[1]>>) outs(%subview_1 : memref<?xi32, strided<[1]>>) {
    ^bb0(%in: f32, %in_2: f32, %out: i32):
      %7 = arith.cmpf oeq, %in, %in_2 : f32
      %8 = arith.extui %7 : i1 to i32
      %9 = arith.ori %out, %8 : i32
      linalg.yield %9 : i32
    }
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%2 : tensor<?xi32>) outs(%0 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?xi32>
    %6 = bufferization.to_memref %5 : memref<?xi32>
    memref.copy %6, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

