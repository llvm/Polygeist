#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nested_all_cpu(%arg0: memref<?x64xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg2 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c1_i32 : i32
    }
    linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : memref<?x64xi32>, memref<?xi32>) outs(%arg2 : memref<?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %0 = linalg.index 1 : index
      %1 = arith.index_cast %in_0 : i32 to index
      %2 = arith.cmpi ult, %0, %1 : index
      %3 = arith.cmpi ne, %in, %c0_i32 : i32
      %4 = arith.extui %3 : i1 to i32
      %5 = arith.andi %out, %4 : i32
      %6 = arith.select %2, %5, %out : i32
      linalg.yield %6 : i32
    }
    return
  }
}

