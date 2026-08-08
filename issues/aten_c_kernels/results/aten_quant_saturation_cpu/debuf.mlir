#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_quant_saturation_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-128_i32 = arith.constant -128 : i32
    %c127_i32 = arith.constant 127 : i32
    %false = arith.constant false
    %0 = bufferization.to_tensor %arg0 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi8>
    %2 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%0 : tensor<?xi32>) outs(%1 : tensor<?xi8>) {
    ^bb0(%in: i32, %out: i8):
      %4 = arith.cmpi slt, %in, %c-128_i32 : i32
      %5 = arith.select %4, %c-128_i32, %in : i32
      %6 = arith.cmpi sgt, %in, %c127_i32 : i32
      %7 = arith.select %4, %false, %6 : i1
      %8 = arith.select %7, %c127_i32, %5 : i32
      %9 = arith.trunci %8 : i32 to i8
      linalg.yield %9 : i8
    } -> tensor<?xi8>
    %3 = bufferization.to_memref %2 : memref<?xi8>
    memref.copy %3, %arg1 : memref<?xi8> to memref<?xi8>
    return
  }
}

