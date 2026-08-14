#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1)[s0] -> (s0, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_int4pack_mm_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x32xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c15_i32 = arith.constant 15 : i32
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c64 = arith.constant 64 : index
    %c48 = arith.constant 48 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg4 : memref<?x48xf32>
    %3 = affine.for %arg5 = 0 to 32 iter_args(%arg6 = %2) -> (tensor<?x48xf32>) {
      %extracted_slice = tensor.extract_slice %arg6[%arg5, 0] [1, %c48] [1, 1] : tensor<?x48xf32> to tensor<?xf32>
      %5 = kernel.launch @memset_zero_1D_f32(%extracted_slice) : (tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg5, 0] [1, %c48] [1, 1] : tensor<?xf32> into tensor<?x48xf32>
      %6 = polygeist.submap(%inserted_slice, %arg5, %c48, %c64) {map = #map1} : (tensor<?x48xf32>, index, index, index) -> tensor<?x?xf32>
      %7 = polygeist.submap(%1, %c48) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
      %8 = polygeist.submap(%7, %c48, %c64) {map = #map2} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
      %9 = polygeist.submap(%0, %c48) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
      %10 = polygeist.submap(%9, %c48, %c64) {map = #map2} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
      %11 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%10, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %13 = linalg.index 0 : index
        %14 = linalg.index 1 : index
        %15 = arith.index_cast %14 : index to i32
        %16 = arith.cmpi slt, %14, %c0 : index
        %17 = arith.subi %c-1, %14 : index
        %18 = arith.select %16, %17, %14 : index
        %19 = arith.divsi %18, %c2 : index
        %20 = arith.subi %c-1, %19 : index
        %21 = arith.select %16, %20, %19 : index
        %22 = memref.load %arg1[%13, %21] : memref<?x32xi8>
        %23 = arith.extui %22 : i8 to i32
        %24 = arith.andi %15, %c1_i32 : i32
        %25 = arith.muli %24, %c4_i32 : i32
        %26 = arith.shrsi %23, %25 : i32
        %27 = arith.andi %26, %c15_i32 : i32
        %28 = memref.load %arg0[%arg5, %14] : memref<?x64xf32>
        %29 = arith.sitofp %27 : i32 to f32
        %30 = arith.subf %29, %in : f32
        %31 = arith.mulf %28, %30 : f32
        %32 = arith.mulf %31, %in_0 : f32
        %33 = arith.addf %out, %32 : f32
        linalg.yield %33 : f32
      } -> tensor<?x?xf32>
      %12 = polygeist.submapInverse(%inserted_slice, %11, %arg5, %c48, %c64) {map = #map1} : (tensor<?x48xf32>, tensor<?x?xf32>, index, index, index) -> tensor<?x48xf32>
      affine.yield %12 : tensor<?x48xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?x48xf32>
    memref.copy %4, %arg4 : memref<?x48xf32> to memref<?x48xf32>
    return
  }
}

