#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_quantized_cpu(%arg0: memref<?xi8>, %arg1: f32, %arg2: i32, %arg3: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c255_i32 = arith.constant 255 : i32
    %false = arith.constant false
    %0 = bufferization.to_tensor %arg0 : memref<?xi8>
    %1 = bufferization.to_tensor %arg3 : memref<?xi8>
    %2 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%0 : tensor<?xi8>) outs(%1 : tensor<?xi8>) {
    ^bb0(%in: i8, %out: i8):
      %4 = arith.extui %in : i8 to i32
      %5 = arith.subi %4, %arg2 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %arg1 : f32
      %8 = arith.divf %7, %arg1 : f32
      %9 = arith.fptosi %8 : f32 to i32
      %10 = arith.addi %9, %arg2 : i32
      %11 = arith.cmpi slt, %10, %c0_i32 : i32
      %12 = arith.select %11, %c0_i32, %10 : i32
      %13 = arith.cmpi sgt, %10, %c255_i32 : i32
      %14 = arith.select %11, %false, %13 : i1
      %15 = arith.select %14, %c255_i32, %12 : i32
      %16 = arith.trunci %15 : i32 to i8
      linalg.yield %16 : i8
    } -> tensor<?xi8>
    %3 = bufferization.to_memref %2 : memref<?xi8>
    memref.copy %3, %arg3 : memref<?xi8> to memref<?xi8>
    return
  }
}

