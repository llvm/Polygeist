#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 36 + d1 + d2 * 12 + d3 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_max_pool3d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c336_i32 = arith.constant 336 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %4 = affine.for %arg3 = 0 to 2 iter_args(%arg4 = %3) -> (tensor<?xf32>) {
      %6 = arith.index_cast %arg3 : index to i32
      %7 = arith.muli %6, %c336_i32 : i32
      %8 = affine.for %arg5 = 0 to 3 iter_args(%arg6 = %arg4) -> (tensor<?xf32>) {
        %9 = affine.for %arg7 = 0 to 3 iter_args(%arg8 = %arg6) -> (tensor<?xf32>) {
          %10 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %arg8) -> (tensor<?xf32>) {
            %11 = affine.apply #map1(%arg3, %arg9, %arg5, %arg7)
            %extracted = tensor.extract %1[%11] : tensor<?xi32>
            %12 = arith.addi %7, %extracted : i32
            %13 = arith.index_cast %12 : i32 to index
            %14 = affine.apply #map1(%arg3, %arg9, %arg5, %arg7)
            %extracted_0 = tensor.extract %2[%14] : tensor<?xf32>
            %extracted_1 = tensor.extract %arg10[%13] : tensor<?xf32>
            %15 = arith.addf %extracted_1, %extracted_0 : f32
            %inserted = tensor.insert %15 into %arg10[%13] : tensor<?xf32>
            affine.yield %inserted : tensor<?xf32>
          }
          affine.yield %10 : tensor<?xf32>
        }
        affine.yield %9 : tensor<?xf32>
      }
      affine.yield %8 : tensor<?xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

