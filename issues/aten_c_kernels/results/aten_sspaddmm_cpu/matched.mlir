#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sspaddmm_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x48xf32>, %arg4: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c48 = arith.constant 48 : index
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x48xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?x48xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c64, %c48] [1, 1] : tensor<?x48xf32> to tensor<?x?xf32>
    %5 = kernel.launch @memset_zero_2D_f32(%extracted_slice) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %5 into %0[0, 0] [%c64, %c48] [1, 1] : tensor<?x?xf32> into tensor<?x48xf32>
    %6 = affine.for %arg5 = 0 to 512 iter_args(%arg6 = %inserted_slice) -> (tensor<?x48xf32>) {
      %8 = affine.for %arg7 = 0 to 48 iter_args(%arg8 = %arg6) -> (tensor<?x48xf32>) {
        %extracted = tensor.extract %4[%arg5] : tensor<?xi32>
        %9 = arith.index_cast %extracted : i32 to index
        %extracted_0 = tensor.extract %2[%arg5] : tensor<?xf32>
        %extracted_1 = tensor.extract %3[%arg5] : tensor<?xi32>
        %10 = arith.index_cast %extracted_1 : i32 to index
        %extracted_2 = tensor.extract %1[%10, %arg7] : tensor<?x48xf32>
        %11 = arith.mulf %extracted_0, %extracted_2 : f32
        %extracted_3 = tensor.extract %arg8[%9, %arg7] : tensor<?x48xf32>
        %12 = arith.addf %extracted_3, %11 : f32
        %inserted = tensor.insert %12 into %arg8[%9, %arg7] : tensor<?x48xf32>
        affine.yield %inserted : tensor<?x48xf32>
      }
      affine.yield %8 : tensor<?x48xf32>
    }
    %7 = bufferization.to_memref %6 : memref<?x48xf32>
    memref.copy %7, %arg4 : memref<?x48xf32> to memref<?x48xf32>
    return
  }
}

