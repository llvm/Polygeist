#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sobol_fast_forward_cpu(%arg0: memref<?xi32>, %arg1: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xi32>
    %1 = polygeist.submap(%0, %c256, %c8) {map = #map} : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["reduction", "parallel"], library_call = ""} outs(%1 : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      %7:2 = scf.while (%arg2 = %6, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %12 = arith.andi %arg2, %c1_i32 : i32
        %13 = arith.cmpi ne, %12, %c0_i32 : i32
        scf.condition(%13) %arg3, %arg2 : i32, i32
      } do {
      ^bb0(%arg2: i32, %arg3: i32):
        %12 = arith.addi %arg2, %c1_i32 : i32
        %13 = arith.shrsi %arg3, %c1_i32 : i32
        scf.yield %13, %12 : i32, i32
      }
      %8 = arith.index_cast %7#0 : i32 to index
      %9 = linalg.index 1 : index
      %10 = memref.load %arg1[%9, %8] : memref<?x32xi32>
      %11 = arith.xori %out, %10 : i32
      linalg.yield %11 : i32
    } -> tensor<?x?xi32>
    %3 = polygeist.submapInverse(%0, %2, %c256, %c8) {map = #map} : (tensor<?xi32>, tensor<?x?xi32>, index, index) -> tensor<?xi32>
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg0 : memref<?xi32> to memref<?xi32>
    return
  }
}

