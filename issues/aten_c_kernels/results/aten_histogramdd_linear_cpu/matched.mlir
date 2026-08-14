#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_histogramdd_linear_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c192_i32 = arith.constant 192 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xi32>
    %3 = kernel.launch @memset_zero_1D_f32(%0) : (tensor<?xf32>) -> tensor<?xf32>
    %4 = affine.for %arg3 = 0 to 4096 iter_args(%arg4 = %3) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg3] : tensor<?xi32>
      %6 = arith.cmpi sge, %extracted, %c0_i32 : i32
      %7 = scf.if %6 -> (tensor<?xf32>) {
        %extracted_0 = tensor.extract %2[%arg3] : tensor<?xi32>
        %8 = arith.cmpi slt, %extracted_0, %c192_i32 : i32
        %9 = scf.if %8 -> (tensor<?xf32>) {
          %extracted_1 = tensor.extract %2[%arg3] : tensor<?xi32>
          %10 = arith.index_cast %extracted_1 : i32 to index
          %extracted_2 = tensor.extract %1[%arg3] : tensor<?xf32>
          %extracted_3 = tensor.extract %arg4[%10] : tensor<?xf32>
          %11 = arith.addf %extracted_3, %extracted_2 : f32
          %inserted = tensor.insert %11 into %arg4[%10] : tensor<?xf32>
          scf.yield %inserted : tensor<?xf32>
        } else {
          scf.yield %arg4 : tensor<?xf32>
        }
        scf.yield %9 : tensor<?xf32>
      } else {
        scf.yield %arg4 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

