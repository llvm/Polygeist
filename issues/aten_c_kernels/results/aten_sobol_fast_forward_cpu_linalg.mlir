#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sobol_fast_forward_cpu(%arg0: memref<?xi32>, %arg1: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c256 = arith.constant 256 : index
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = polygeist.submap(%arg0, %c256, %c8) {map = #map} : (memref<?xi32>, index, index) -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["reduction", "parallel"]} outs(%0 : memref<?x?xi32>) {
    ^bb0(%out: i32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3:2 = scf.while (%arg2 = %2, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %8 = arith.andi %arg2, %c1_i32 : i32
        %9 = arith.cmpi ne, %8, %c0_i32 : i32
        scf.condition(%9) %arg3, %arg2 : i32, i32
      } do {
      ^bb0(%arg2: i32, %arg3: i32):
        %8 = arith.addi %arg2, %c1_i32 : i32
        %9 = arith.shrsi %arg3, %c1_i32 : i32
        scf.yield %9, %8 : i32, i32
      }
      %4 = arith.index_cast %3#0 : i32 to index
      %5 = linalg.index 1 : index
      %6 = memref.load %arg1[%5, %4] : memref<?x32xi32>
      %7 = arith.xori %out, %6 : i32
      linalg.yield %7 : i32
    }
    return
  }
}

