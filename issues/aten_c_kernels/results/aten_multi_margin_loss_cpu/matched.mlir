#map = affine_map<(d0)[s0] -> (s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multi_margin_loss_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.600000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg5 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xi32>
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = tensor.empty(%c32) : tensor<?xf32>
    %5:2 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %4, %arg8 = %0) -> (tensor<?xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg6] : tensor<?xi32>
      %7 = arith.index_cast %extracted : i32 to index
      %inserted = tensor.insert %cst_0 into %arg7[%arg6] : tensor<?xf32>
      %8 = polygeist.submap(%inserted, %arg6, %c16) {map = #map} : (tensor<?xf32>, index, index) -> tensor<?xf32>
      %9 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["reduction"], library_call = ""} outs(%8 : tensor<?xf32>) {
      ^bb0(%out: f32):
        %13 = linalg.index 0 : index
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.cmpi ne, %14, %extracted : i32
        %16 = scf.if %15 -> (f32) {
          %17 = memref.load %arg0[%arg6, %7] : memref<?x16xf32>
          %18 = arith.subf %arg3, %17 : f32
          %19 = memref.load %arg0[%arg6, %13] : memref<?x16xf32>
          %20 = arith.addf %18, %19 : f32
          %21 = arith.cmpf ogt, %20, %cst_0 : f32
          %22 = scf.if %21 -> (f32) {
            %23 = affine.apply #map2()[%3]
            %24 = arith.cmpi eq, %23, %c0 : index
            %25 = arith.mulf %20, %20 : f32
            %26 = arith.select %24, %20, %25 : f32
            %27 = arith.addf %out, %26 : f32
            scf.yield %27 : f32
          } else {
            scf.yield %out : f32
          }
          scf.yield %22 : f32
        } else {
          scf.yield %out : f32
        }
        linalg.yield %16 : f32
      } -> tensor<?xf32>
      %10 = polygeist.submapInverse(%inserted, %9, %arg6, %c16) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index, index) -> tensor<?xf32>
      %extracted_1 = tensor.extract %10[%arg6] : tensor<?xf32>
      %extracted_2 = tensor.extract %1[%7] : tensor<?xf32>
      %11 = arith.mulf %extracted_1, %extracted_2 : f32
      %12 = arith.divf %11, %cst : f32
      %inserted_3 = tensor.insert %12 into %arg8[%arg6] : tensor<?xf32>
      affine.yield %10, %inserted_3 : tensor<?xf32>, tensor<?xf32>
    }
    %6 = bufferization.to_memref %5#1 : memref<?xf32>
    memref.copy %6, %arg5 : memref<?xf32> to memref<?xf32>
    return
  }
}

