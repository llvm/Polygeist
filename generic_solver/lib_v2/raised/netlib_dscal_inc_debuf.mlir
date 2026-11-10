#map = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dscal_inc(%arg0: i32, %arg1: f64, %arg2: memref<?xf64>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = arith.muli %arg0, %arg3 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %5 = arith.addi %4, %2 : index
    %6 = arith.divui %5, %3 : index
    %7 = polygeist.submap(%0, %3, %6) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%7 : tensor<?xf64>) {
    ^bb0(%out: f64):
      %11 = arith.mulf %arg1, %out : f64
      linalg.yield %11 : f64
    } -> tensor<?xf64>
    %9 = polygeist.submapInverse(%0, %8, %3, %6) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
    %10 = bufferization.to_memref %9 : memref<?xf64>
    memref.copy %10, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}

