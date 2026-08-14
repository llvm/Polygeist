#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sort_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?x64xi32>
    %2 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %3 = kernel.launch @cudaCopy2D_f32_tensor(%extracted_slice_0, %extracted_slice) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %2[0, 0] [%c16, %c64] [1, 1] : tensor<?x?xf32> into tensor<?x64xf32>
    %extracted_slice_1 = tensor.extract_slice %1[0, 0] [%c16, %c64] [1, 1] : tensor<?x64xi32> to tensor<?x?xi32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice_1 : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      %8 = linalg.index 1 : index
      %9 = arith.index_cast %8 : index to i32
      linalg.yield %9 : i32
    } -> tensor<?x?xi32>
    %inserted_slice_2 = tensor.insert_slice %4 into %1[0, 0] [%c16, %c64] [1, 1] : tensor<?x?xi32> into tensor<?x64xi32>
    %5:2 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %inserted_slice, %arg5 = %inserted_slice_2) -> (tensor<?x64xf32>, tensor<?x64xi32>) {
      %8:2 = affine.for %arg6 = 1 to 64 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?x64xf32>, tensor<?x64xi32>) {
        %9 = arith.index_cast %arg6 : index to i32
        %extracted = tensor.extract %arg7[%arg3, %arg6] : tensor<?x64xf32>
        %extracted_3 = tensor.extract %arg8[%arg3, %arg6] : tensor<?x64xi32>
        %10 = arith.addi %9, %c-1_i32 : i32
        %11:3 = scf.while (%arg9 = %10, %arg10 = %arg7, %arg11 = %arg8) : (i32, tensor<?x64xf32>, tensor<?x64xi32>) -> (i32, tensor<?x64xf32>, tensor<?x64xi32>) {
          %14 = arith.cmpi sge, %arg9, %c0_i32 : i32
          %15:4 = scf.if %14 -> (i1, i32, tensor<?x64xf32>, tensor<?x64xi32>) {
            %16 = arith.index_cast %arg9 : i32 to index
            %extracted_5 = tensor.extract %arg10[%arg3, %16] : tensor<?x64xf32>
            %17 = arith.cmpf olt, %extracted_5, %extracted : f32
            %18:3 = scf.if %17 -> (i32, tensor<?x64xf32>, tensor<?x64xi32>) {
              %19 = arith.addi %arg9, %c1_i32 : i32
              %20 = arith.index_cast %19 : i32 to index
              %extracted_6 = tensor.extract %arg10[%arg3, %16] : tensor<?x64xf32>
              %inserted_7 = tensor.insert %extracted_6 into %arg10[%arg3, %20] : tensor<?x64xf32>
              %extracted_8 = tensor.extract %arg11[%arg3, %16] : tensor<?x64xi32>
              %inserted_9 = tensor.insert %extracted_8 into %arg11[%arg3, %20] : tensor<?x64xi32>
              %21 = arith.addi %arg9, %c-1_i32 : i32
              scf.yield %21, %inserted_7, %inserted_9 : i32, tensor<?x64xf32>, tensor<?x64xi32>
            } else {
              scf.yield %arg9, %arg10, %arg11 : i32, tensor<?x64xf32>, tensor<?x64xi32>
            }
            scf.yield %17, %18#0, %18#1, %18#2 : i1, i32, tensor<?x64xf32>, tensor<?x64xi32>
          } else {
            scf.yield %false, %arg9, %arg10, %arg11 : i1, i32, tensor<?x64xf32>, tensor<?x64xi32>
          }
          scf.condition(%15#0) %15#1, %15#2, %15#3 : i32, tensor<?x64xf32>, tensor<?x64xi32>
        } do {
        ^bb0(%arg9: i32, %arg10: tensor<?x64xf32>, %arg11: tensor<?x64xi32>):
          scf.yield %arg9, %arg10, %arg11 : i32, tensor<?x64xf32>, tensor<?x64xi32>
        }
        %12 = arith.addi %11#0, %c1_i32 : i32
        %13 = arith.index_cast %12 : i32 to index
        %inserted = tensor.insert %extracted into %11#1[%arg3, %13] : tensor<?x64xf32>
        %inserted_4 = tensor.insert %extracted_3 into %11#2[%arg3, %13] : tensor<?x64xi32>
        affine.yield %inserted, %inserted_4 : tensor<?x64xf32>, tensor<?x64xi32>
      }
      affine.yield %8#0, %8#1 : tensor<?x64xf32>, tensor<?x64xi32>
    }
    %6 = bufferization.to_memref %5#1 : memref<?x64xi32>
    memref.copy %6, %arg2 : memref<?x64xi32> to memref<?x64xi32>
    %7 = bufferization.to_memref %5#0 : memref<?x64xf32>
    memref.copy %7, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

