#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sampled_addmm_sparse_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x32xf32>, %arg4: memref<?x24xf32>, %arg5: f32, %arg6: f32, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg7 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xi32>
    %3 = bufferization.to_tensor %arg0 : memref<?xi32>
    %4 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %3[%arg8] : tensor<?xi32>
      %6:2 = scf.while (%arg10 = %extracted, %arg11 = %arg9) : (i32, tensor<?xf32>) -> (i32, tensor<?xf32>) {
        %7 = affine.apply #map(%arg8)
        %extracted_0 = tensor.extract %3[%7] : tensor<?xi32>
        %8 = arith.cmpi slt, %arg10, %extracted_0 : i32
        scf.condition(%8) %arg10, %arg11 : i32, tensor<?xf32>
      } do {
      ^bb0(%arg10: i32, %arg11: tensor<?xf32>):
        %7 = arith.index_cast %arg10 : i32 to index
        %extracted_0 = tensor.extract %2[%7] : tensor<?xi32>
        %8 = arith.index_cast %extracted_0 : i32 to index
        %alloca = memref.alloca() : memref<f32>
        %9 = bufferization.to_tensor %alloca : memref<f32>
        %inserted = tensor.insert %cst into %9[] : tensor<f32>
        %10 = polygeist.submap(%inserted, %c32) {map = #map1} : (tensor<f32>, index) -> tensor<?xf32>
        %11 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["reduction"], library_call = ""} outs(%10 : tensor<?xf32>) {
        ^bb0(%out: f32):
          %17 = linalg.index 0 : index
          %18 = memref.load %arg3[%arg8, %17] : memref<?x32xf32>
          %19 = memref.load %arg4[%17, %8] : memref<?x24xf32>
          %20 = arith.mulf %18, %19 : f32
          %21 = arith.addf %out, %20 : f32
          linalg.yield %21 : f32
        } -> tensor<?xf32>
        %12 = polygeist.submapInverse(%inserted, %11, %c32) {map = #map1} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
        %extracted_1 = tensor.extract %12[] : tensor<f32>
        %extracted_2 = tensor.extract %1[%7] : tensor<?xf32>
        %13 = arith.mulf %arg6, %extracted_2 : f32
        %14 = arith.mulf %arg5, %extracted_1 : f32
        %15 = arith.addf %13, %14 : f32
        %inserted_3 = tensor.insert %15 into %arg11[%7] : tensor<?xf32>
        %16 = arith.addi %arg10, %c1_i32 : i32
        scf.yield %16, %inserted_3 : i32, tensor<?xf32>
      }
      affine.yield %6#1 : tensor<?xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg7 : memref<?xf32> to memref<?xf32>
    return
  }
}

