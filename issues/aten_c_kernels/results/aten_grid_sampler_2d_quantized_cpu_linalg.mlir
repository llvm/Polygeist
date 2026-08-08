#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_quantized_cpu(%arg0: memref<?xi8>, %arg1: f32, %arg2: i32, %arg3: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %c255_i32 = arith.constant 255 : i32
    %c0_i32 = arith.constant 0 : i32
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xi8>) outs(%arg3 : memref<?xi8>) {
    ^bb0(%in: i8, %out: i8):
      %0 = arith.extui %in : i8 to i32
      %1 = arith.subi %0, %arg2 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %arg1 : f32
      %4 = arith.divf %3, %arg1 : f32
      %5 = arith.fptosi %4 : f32 to i32
      %6 = arith.addi %5, %arg2 : i32
      %7 = arith.cmpi slt, %6, %c0_i32 : i32
      %8 = arith.select %7, %c0_i32, %6 : i32
      %9 = arith.cmpi sgt, %6, %c255_i32 : i32
      %10 = arith.select %7, %false, %9 : i1
      %11 = arith.select %10, %c255_i32, %8 : i32
      %12 = arith.trunci %11 : i32 to i8
      linalg.yield %12 : i8
    }
    return
  }
}

