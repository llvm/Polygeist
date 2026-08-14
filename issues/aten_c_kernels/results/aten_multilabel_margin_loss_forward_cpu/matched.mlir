#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multilabel_margin_loss_forward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x4xi32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 1.600000e+01 : f32
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x4xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %3 = llvm.mlir.undef : f32
    %4 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %3 into %4[] : tensor<f32>
    %5 = tensor.empty(%c16) : tensor<?xf32>
    %6 = tensor.empty(%c16) : tensor<?xf32>
    %7:4 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %inserted, %arg5 = %5, %arg6 = %6, %arg7 = %0) -> (tensor<f32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg4[] : tensor<f32>
      %inserted_2 = tensor.insert %extracted into %arg5[%arg3] : tensor<?xf32>
      %inserted_3 = tensor.insert %cst into %arg6[%arg3] : tensor<?xf32>
      %alloca = memref.alloca(%c4) : memref<?xf32>
      %9 = bufferization.to_tensor %alloca : memref<?xf32>
      %alloca_4 = memref.alloca(%c4) : memref<?xf32>
      %10 = bufferization.to_tensor %alloca_4 : memref<?xf32>
      %11:4 = affine.for %arg8 = 0 to 4 iter_args(%arg9 = %inserted_2, %arg10 = %inserted_3, %arg11 = %9, %arg12 = %10) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
        %extracted_9 = tensor.extract %arg9[%arg3] : tensor<?xf32>
        %extracted_10 = tensor.extract %arg10[%arg3] : tensor<?xf32>
        %extracted_11 = tensor.extract %1[%arg3, %arg8] : tensor<?x4xi32>
        %13 = arith.index_cast %extracted_11 : i32 to index
        %inserted_12 = tensor.insert %extracted_9 into %arg11[%arg8] : tensor<?xf32>
        %inserted_13 = tensor.insert %extracted_10 into %arg12[%arg8] : tensor<?xf32>
        %alloca_14 = memref.alloca(%c16) : memref<?xi32>
        %14 = bufferization.to_tensor %alloca_14 : memref<?xi32>
        %15:3 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %inserted_12, %arg15 = %inserted_13, %arg16 = %14) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xi32>) {
          %extracted_19 = tensor.extract %arg14[%arg8] : tensor<?xf32>
          %extracted_20 = tensor.extract %arg15[%arg8] : tensor<?xf32>
          %16 = arith.index_cast %arg13 : index to i32
          %inserted_21 = tensor.insert %c0_i32 into %arg16[%arg13] : tensor<?xi32>
          %extracted_slice = tensor.extract_slice %inserted_21[%arg13] [1] [1] : tensor<?xi32> to tensor<i32>
          %extracted_slice_22 = tensor.extract_slice %1[%arg3, 0] [1, %c4] [1, 1] : tensor<?x4xi32> to tensor<?xi32>
          %17 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_22 : tensor<?xi32>) outs(%extracted_slice : tensor<i32>) {
          ^bb0(%in: i32, %out: i32):
            %26 = arith.cmpi eq, %in, %16 : i32
            %27 = arith.extui %26 : i1 to i32
            %28 = arith.ori %out, %27 : i32
            linalg.yield %28 : i32
          } -> tensor<i32>
          %inserted_slice = tensor.insert_slice %17 into %inserted_21[%arg13] [1] [1] : tensor<i32> into tensor<?xi32>
          %extracted_23 = tensor.extract %inserted_slice[%arg13] : tensor<?xi32>
          %18 = arith.cmpi eq, %extracted_23, %c0_i32 : i32
          %extracted_24 = tensor.extract %2[%arg3, %13] : tensor<?x16xf32>
          %19 = arith.subf %cst_0, %extracted_24 : f32
          %extracted_25 = tensor.extract %2[%arg3, %arg13] : tensor<?x16xf32>
          %20 = arith.addf %19, %extracted_25 : f32
          %21 = arith.cmpf ogt, %20, %cst : f32
          %22 = arith.addf %extracted_20, %20 : f32
          %23 = arith.select %21, %22, %extracted_20 : f32
          %24 = arith.select %18, %20, %extracted_19 : f32
          %25 = arith.select %18, %23, %extracted_20 : f32
          %inserted_26 = tensor.insert %24 into %arg14[%arg8] : tensor<?xf32>
          %inserted_27 = tensor.insert %25 into %arg15[%arg8] : tensor<?xf32>
          affine.yield %inserted_26, %inserted_27, %inserted_slice : tensor<?xf32>, tensor<?xf32>, tensor<?xi32>
        }
        %extracted_15 = tensor.extract %15#0[%arg8] : tensor<?xf32>
        %extracted_16 = tensor.extract %15#1[%arg8] : tensor<?xf32>
        %inserted_17 = tensor.insert %extracted_15 into %arg9[%arg3] : tensor<?xf32>
        %inserted_18 = tensor.insert %extracted_16 into %arg10[%arg3] : tensor<?xf32>
        affine.yield %inserted_17, %inserted_18, %15#0, %15#1 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
      }
      %extracted_5 = tensor.extract %11#0[%arg3] : tensor<?xf32>
      %extracted_6 = tensor.extract %11#1[%arg3] : tensor<?xf32>
      %12 = arith.divf %extracted_6, %cst_1 : f32
      %inserted_7 = tensor.insert %12 into %arg7[%arg3] : tensor<?xf32>
      %inserted_8 = tensor.insert %extracted_5 into %arg4[] : tensor<f32>
      affine.yield %inserted_8, %11#0, %11#1, %inserted_7 : tensor<f32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
    }
    %8 = bufferization.to_memref %7#3 : memref<?xf32>
    memref.copy %8, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

