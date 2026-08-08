#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_csr_reduce_dim0_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xi32>
    %3 = bufferization.to_tensor %arg0 : memref<?xi32>
    %4 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %5 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %4) -> (tensor<?xf32>) {
      %extracted = tensor.extract %3[%arg4] : tensor<?xi32>
      %7:2 = scf.while (%arg6 = %extracted, %arg7 = %arg5) : (i32, tensor<?xf32>) -> (i32, tensor<?xf32>) {
        %8 = affine.apply #map1(%arg4)
        %extracted_0 = tensor.extract %3[%8] : tensor<?xi32>
        %9 = arith.cmpi slt, %arg6, %extracted_0 : i32
        scf.condition(%9) %arg6, %arg7 : i32, tensor<?xf32>
      } do {
      ^bb0(%arg6: i32, %arg7: tensor<?xf32>):
        %8 = arith.index_cast %arg6 : i32 to index
        %extracted_0 = tensor.extract %2[%8] : tensor<?xi32>
        %9 = arith.index_cast %extracted_0 : i32 to index
        %extracted_1 = tensor.extract %1[%8] : tensor<?xf32>
        %extracted_2 = tensor.extract %arg7[%9] : tensor<?xf32>
        %10 = arith.addf %extracted_2, %extracted_1 : f32
        %inserted = tensor.insert %10 into %arg7[%9] : tensor<?xf32>
        %11 = arith.addi %arg6, %c1_i32 : i32
        scf.yield %11, %inserted : i32, tensor<?xf32>
      }
      affine.yield %7#1 : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

