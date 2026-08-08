#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fused_adagrad_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = arith.subf %arg7, %cst : f32
    %4 = arith.mulf %3, %arg4 : f32
    %5 = arith.addf %4, %cst : f32
    %6 = arith.divf %arg3, %5 : f32
    %7 = arith.cmpi ne, %arg9, %c0_i32 : i32
    %8 = arith.cmpf une, %arg5, %cst_0 : f32
    %9:3 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} outs(%1, %2, %0 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%out: f32, %out_1: f32, %out_2: f32):
      %13 = arith.divf %out, %arg8 : f32
      %14 = arith.negf %13 : f32
      %15 = arith.select %7, %14, %13 : f32
      %16 = arith.mulf %out_2, %arg5 : f32
      %17 = arith.addf %15, %16 : f32
      %18 = arith.select %8, %17, %15 : f32
      %19 = arith.mulf %18, %18 : f32
      %20 = arith.addf %out_1, %19 : f32
      %21 = arith.mulf %6, %18 : f32
      %22 = math.sqrt %20 : f32
      %23 = arith.addf %22, %arg6 : f32
      %24 = arith.divf %21, %23 : f32
      %25 = arith.subf %out_2, %24 : f32
      linalg.yield %13, %20, %25 : f32, f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
    %10 = bufferization.to_memref %9#2 : memref<?xf32>
    memref.copy %10, %arg0 : memref<?xf32> to memref<?xf32>
    %11 = bufferization.to_memref %9#0 : memref<?xf32>
    memref.copy %11, %arg1 : memref<?xf32> to memref<?xf32>
    %12 = bufferization.to_memref %9#1 : memref<?xf32>
    memref.copy %12, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

