#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_backward_input_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?x24xf32>, %arg3: memref<?x24xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c24 = arith.constant 24 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xi32>
    %3 = affine.for %arg5 = 0 to 16 iter_args(%arg6 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg5] : tensor<?xi32>
      %5:2 = scf.while (%arg7 = %extracted, %arg8 = %arg6) : (i32, tensor<?xf32>) -> (i32, tensor<?xf32>) {
        %6 = affine.apply #map(%arg5)
        %extracted_0 = tensor.extract %2[%6] : tensor<?xi32>
        %7 = arith.cmpi slt, %arg7, %extracted_0 : i32
        scf.condition(%7) %arg7, %arg8 : i32, tensor<?xf32>
      } do {
      ^bb0(%arg7: i32, %arg8: tensor<?xf32>):
        %6 = arith.index_cast %arg7 : i32 to index
        %extracted_0 = tensor.extract %1[%6] : tensor<?xi32>
        %7 = arith.index_cast %extracted_0 : i32 to index
        %alloca = memref.alloca() : memref<f32>
        %8 = bufferization.to_tensor %alloca : memref<f32>
        %inserted = tensor.insert %cst into %8[] : tensor<f32>
        %9 = polygeist.submap(%inserted, %c24) {map = #map1} : (tensor<f32>, index) -> tensor<?xf32>
        %10 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["reduction"], library_call = ""} outs(%9 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %13 = linalg.index 0 : index
          %14 = memref.load %arg2[%arg5, %13] : memref<?x24xf32>
          %15 = memref.load %arg3[%7, %13] : memref<?x24xf32>
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %out, %16 : f32
          linalg.yield %17 : f32
        } -> tensor<?xf32>
        %11 = polygeist.submapInverse(%inserted, %10, %c24) {map = #map1} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
        %extracted_1 = tensor.extract %11[] : tensor<f32>
        %inserted_2 = tensor.insert %extracted_1 into %arg8[%6] : tensor<?xf32>
        %12 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %12, %inserted_2 : i32, tensor<?xf32>
      }
      affine.yield %5#1 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg4 : memref<?xf32> to memref<?xf32>
    return
  }
}

