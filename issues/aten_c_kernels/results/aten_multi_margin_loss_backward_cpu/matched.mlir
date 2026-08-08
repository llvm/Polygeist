#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multi_margin_loss_backward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: i32, %arg5: memref<?xf32>, %arg6: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.600000e+01 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x16xf32>
    %1 = bufferization.to_tensor %arg5 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = tensor.empty(%c32) : tensor<?xf32>
    %7:2 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %6, %arg9 = %0) -> (tensor<?xf32>, tensor<?x16xf32>) {
      %extracted = tensor.extract %3[%arg7] : tensor<?xi32>
      %9 = arith.index_cast %extracted : i32 to index
      %inserted = tensor.insert %cst_2 into %arg8[%arg7] : tensor<?xf32>
      %extracted_slice = tensor.extract_slice %arg9[%arg7, 0] [1, %c16] [1, 1] : tensor<?x16xf32> to tensor<?xf32>
      %10 = kernel.launch @memset_zero_1D_f32(%extracted_slice) : (tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %10 into %arg9[%arg7, 0] [1, %c16] [1, 1] : tensor<?xf32> into tensor<?x16xf32>
      %11:2 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %inserted, %arg12 = %inserted_slice) -> (tensor<?xf32>, tensor<?x16xf32>) {
        %extracted_5 = tensor.extract %arg11[%arg7] : tensor<?xf32>
        %13 = arith.index_cast %arg10 : index to i32
        %14 = arith.cmpi ne, %13, %extracted : i32
        %15:2 = scf.if %14 -> (f32, tensor<?x16xf32>) {
          %extracted_7 = tensor.extract %4[%arg7, %9] : tensor<?x16xf32>
          %16 = arith.subf %arg3, %extracted_7 : f32
          %extracted_8 = tensor.extract %4[%arg7, %arg10] : tensor<?x16xf32>
          %17 = arith.addf %16, %extracted_8 : f32
          %18 = arith.cmpf ogt, %17, %cst_2 : f32
          %19:2 = scf.if %18 -> (f32, tensor<?x16xf32>) {
            %extracted_9 = tensor.extract %1[%arg7] : tensor<?xf32>
            %extracted_10 = tensor.extract %2[%9] : tensor<?xf32>
            %20 = arith.mulf %extracted_9, %extracted_10 : f32
            %21 = affine.apply #map1()[%5]
            %22 = arith.cmpi eq, %21, %c0 : index
            %23 = arith.mulf %17, %cst_0 : f32
            %24 = arith.select %22, %cst_1, %23 : f32
            %25 = arith.mulf %20, %24 : f32
            %26 = arith.divf %25, %cst : f32
            %inserted_11 = tensor.insert %26 into %arg12[%arg7, %arg10] : tensor<?x16xf32>
            %27 = arith.addf %extracted_5, %26 : f32
            scf.yield %27, %inserted_11 : f32, tensor<?x16xf32>
          } else {
            scf.yield %extracted_5, %arg12 : f32, tensor<?x16xf32>
          }
          scf.yield %19#0, %19#1 : f32, tensor<?x16xf32>
        } else {
          scf.yield %extracted_5, %arg12 : f32, tensor<?x16xf32>
        }
        %inserted_6 = tensor.insert %15#0 into %arg11[%arg7] : tensor<?xf32>
        affine.yield %inserted_6, %15#1 : tensor<?xf32>, tensor<?x16xf32>
      }
      %extracted_3 = tensor.extract %11#0[%arg7] : tensor<?xf32>
      %12 = arith.negf %extracted_3 : f32
      %inserted_4 = tensor.insert %12 into %11#1[%arg7, %9] : tensor<?x16xf32>
      affine.yield %11#0, %inserted_4 : tensor<?xf32>, tensor<?x16xf32>
    }
    %8 = bufferization.to_memref %7#1 : memref<?x16xf32>
    memref.copy %8, %arg6 : memref<?x16xf32> to memref<?x16xf32>
    return
  }
}

