#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_gcd_i32(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) outs(%arg2 : memref<?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %0 = arith.cmpi slt, %in, %c0_i32 : i32
      %1 = arith.subi %c0_i32, %in : i32
      %2 = arith.select %0, %1, %in : i32
      %3 = arith.cmpi slt, %in_0, %c0_i32 : i32
      %4 = arith.subi %c0_i32, %in_0 : i32
      %5 = arith.select %3, %4, %in_0 : i32
      %6:2 = scf.while (%arg3 = %5, %arg4 = %2) : (i32, i32) -> (i32, i32) {
        %7 = arith.cmpi ne, %arg3, %c0_i32 : i32
        scf.condition(%7) %arg4, %arg3 : i32, i32
      } do {
      ^bb0(%arg3: i32, %arg4: i32):
        %7 = arith.remsi %arg3, %arg4 : i32
        scf.yield %7, %arg4 : i32, i32
      }
      linalg.yield %6#0 : i32
    }
    return
  }
}

