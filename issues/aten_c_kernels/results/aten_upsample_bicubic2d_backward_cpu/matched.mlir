#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_backward_cpu(%arg0: memref<?x8xf32>, %arg1: memref<?x5xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c5 = arith.constant 5 : index
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c25 = arith.constant 25 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x5xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x8xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [1, %c25] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
    %2 = kernel.launch @memset_zero_1D_f32(%extracted_slice) : (tensor<?xf32>) -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %2 into %0[0, 0] [1, %c25] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
    %3 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %inserted_slice) -> (tensor<?x5xf32>) {
      %5 = arith.muli %arg2, %c5 : index
      %6 = arith.cmpi slt, %5, %c0 : index
      %7 = arith.subi %c-1, %5 : index
      %8 = arith.select %6, %7, %5 : index
      %9 = arith.divsi %8, %c8 : index
      %10 = arith.subi %c-1, %9 : index
      %11 = arith.select %6, %10, %9 : index
      %12 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %arg3) -> (tensor<?x5xf32>) {
        %extracted = tensor.extract %1[%arg2, %arg4] : tensor<?x8xf32>
        %13 = arith.muli %arg4, %c5 : index
        %14 = arith.cmpi slt, %13, %c0 : index
        %15 = arith.subi %c-1, %13 : index
        %16 = arith.select %14, %15, %13 : index
        %17 = arith.divsi %16, %c8 : index
        %18 = arith.subi %c-1, %17 : index
        %19 = arith.select %14, %18, %17 : index
        %extracted_0 = tensor.extract %arg5[%11, %19] : tensor<?x5xf32>
        %20 = arith.addf %extracted_0, %extracted : f32
        %inserted = tensor.insert %20 into %arg5[%11, %19] : tensor<?x5xf32>
        affine.yield %inserted : tensor<?x5xf32>
      }
      affine.yield %12 : tensor<?x5xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?x5xf32>
    memref.copy %4, %arg1 : memref<?x5xf32> to memref<?x5xf32>
    return
  }
}

