#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multilabel_margin_loss_backward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?x4xi32>, %arg2: memref<?xf32>, %arg3: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 1.600000e+01 : f32
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x16xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?x4xi32>
    %3 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %4 = llvm.mlir.undef : f32
    %5 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %4 into %5[] : tensor<f32>
    %6:2 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %inserted, %arg6 = %0) -> (tensor<f32>, tensor<?x16xf32>) {
      %extracted = tensor.extract %arg5[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %arg6[%arg4, 0] [1, %c16] [1, 1] : tensor<?x16xf32> to tensor<?xf32>
      %8 = kernel.launch @memset_zero_1D_f32(%extracted_slice) : (tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg6[%arg4, 0] [1, %c16] [1, 1] : tensor<?xf32> into tensor<?x16xf32>
      %inserted_2 = tensor.insert %extracted into %arg5[] : tensor<f32>
      %9:2 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %inserted_2, %arg9 = %inserted_slice) -> (tensor<f32>, tensor<?x16xf32>) {
        %extracted_3 = tensor.extract %2[%arg4, %arg7] : tensor<?x4xi32>
        %10 = arith.index_cast %extracted_3 : i32 to index
        %alloca = memref.alloca(%c16) : memref<?xi32>
        %11 = bufferization.to_tensor %alloca : memref<?xi32>
        %12 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%11 : tensor<?xi32>) {
        ^bb0(%out: i32):
          linalg.yield %c0_i32 : i32
        } -> tensor<?xi32>
        %extracted_slice_4 = tensor.extract_slice %12[0] [%c16] [1] : tensor<?xi32> to tensor<?xi32>
        %extracted_slice_5 = tensor.extract_slice %2[%arg4, 0] [1, %c4] [1, 1] : tensor<?x4xi32> to tensor<?xi32>
        %13 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice_5 : tensor<?xi32>) outs(%extracted_slice_4 : tensor<?xi32>) {
        ^bb0(%in: i32, %out: i32):
          %15 = linalg.index 0 : index
          %16 = arith.index_cast %15 : index to i32
          %17 = arith.cmpi eq, %in, %16 : i32
          %18 = arith.extui %17 : i1 to i32
          %19 = arith.ori %out, %18 : i32
          linalg.yield %19 : i32
        } -> tensor<?xi32>
        %inserted_slice_6 = tensor.insert_slice %13 into %12[0] [%c16] [1] : tensor<?xi32> into tensor<?xi32>
        %14:2 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<f32>, tensor<?x16xf32>) {
          %extracted_7 = tensor.extract %arg11[] : tensor<f32>
          %extracted_8 = tensor.extract %inserted_slice_6[%arg10] : tensor<?xi32>
          %15 = arith.cmpi ne, %extracted_8, %c0_i32 : i32
          %16:2 = scf.if %15 -> (f32, tensor<?x16xf32>) {
            scf.yield %extracted_7, %arg12 : f32, tensor<?x16xf32>
          } else {
            %extracted_10 = tensor.extract %3[%arg4, %10] : tensor<?x16xf32>
            %17 = arith.subf %cst_0, %extracted_10 : f32
            %extracted_11 = tensor.extract %3[%arg4, %arg10] : tensor<?x16xf32>
            %18 = arith.addf %17, %extracted_11 : f32
            %19 = arith.cmpf ogt, %18, %cst : f32
            %20:2 = scf.if %19 -> (f32, tensor<?x16xf32>) {
              %extracted_12 = tensor.extract %1[%arg4] : tensor<?xf32>
              %21 = arith.divf %extracted_12, %cst_1 : f32
              %extracted_13 = tensor.extract %arg12[%arg4, %arg10] : tensor<?x16xf32>
              %22 = arith.addf %extracted_13, %21 : f32
              %inserted_14 = tensor.insert %22 into %arg12[%arg4, %arg10] : tensor<?x16xf32>
              %extracted_15 = tensor.extract %inserted_14[%arg4, %10] : tensor<?x16xf32>
              %23 = arith.subf %extracted_15, %21 : f32
              %inserted_16 = tensor.insert %23 into %inserted_14[%arg4, %10] : tensor<?x16xf32>
              scf.yield %21, %inserted_16 : f32, tensor<?x16xf32>
            } else {
              scf.yield %extracted_7, %arg12 : f32, tensor<?x16xf32>
            }
            scf.yield %20#0, %20#1 : f32, tensor<?x16xf32>
          }
          %inserted_9 = tensor.insert %16#0 into %arg11[] : tensor<f32>
          affine.yield %inserted_9, %16#1 : tensor<f32>, tensor<?x16xf32>
        }
        affine.yield %14#0, %14#1 : tensor<f32>, tensor<?x16xf32>
      }
      affine.yield %9#0, %9#1 : tensor<f32>, tensor<?x16xf32>
    }
    %7 = bufferization.to_memref %6#1 : memref<?x16xf32>
    memref.copy %7, %arg3 : memref<?x16xf32> to memref<?x16xf32>
    return
  }
}

