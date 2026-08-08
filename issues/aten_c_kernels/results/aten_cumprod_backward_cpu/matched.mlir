#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cumprod_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %4 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %3) -> (tensor<?xf32>) {
      %6 = arith.index_cast %arg4 : index to i32
      %7 = affine.for %arg6 = #map(%arg4) to 128 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
        %extracted = tensor.extract %arg7[%arg4] : tensor<?xf32>
        %alloca = memref.alloca() : memref<f32>
        %8 = bufferization.to_tensor %alloca : memref<f32>
        %inserted = tensor.insert %cst_0 into %8[] : tensor<f32>
        %9 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%2 : tensor<?xf32>) outs(%inserted : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %12 = linalg.index 0 : index
          %13 = arith.index_cast %12 : index to i32
          %14 = arith.cmpi ne, %13, %6 : i32
          %15 = arith.mulf %out, %in : f32
          %16 = arith.select %14, %15, %out : f32
          %17 = linalg.index 0 : index
          %18 = affine.apply #map2(%arg6)
          %19 = arith.cmpi slt, %17, %18 : index
          %20 = arith.select %19, %16, %out : f32
          linalg.yield %20 : f32
        } -> tensor<f32>
        %extracted_1 = tensor.extract %9[] : tensor<f32>
        %extracted_2 = tensor.extract %1[%arg6] : tensor<?xf32>
        %10 = arith.mulf %extracted_2, %extracted_1 : f32
        %11 = arith.addf %extracted, %10 : f32
        %inserted_3 = tensor.insert %11 into %arg7[%arg4] : tensor<?xf32>
        affine.yield %inserted_3 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

