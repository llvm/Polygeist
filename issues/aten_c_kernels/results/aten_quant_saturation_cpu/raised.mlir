#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_quant_saturation_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %c127_i32 = arith.constant 127 : i32
    %c-128_i32 = arith.constant -128 : i32
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xi32>) outs(%arg1 : memref<?xi8>) {
    ^bb0(%in: i32, %out: i8):
      %0 = arith.cmpi slt, %in, %c-128_i32 : i32
      %1 = arith.select %0, %c-128_i32, %in : i32
      %2 = arith.cmpi sgt, %in, %c127_i32 : i32
      %3 = arith.select %0, %false, %2 : i1
      %4 = arith.select %3, %c127_i32, %1 : i32
      %5 = arith.trunci %4 : i32 to i8
      linalg.yield %5 : i8
    }
    return
  }
}

