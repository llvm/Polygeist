#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spdiags_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c16, %c16] [1, 1] : tensor<?x16xf32> to tensor<?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %0[0, 0] [%c16, %c16] [1, 1] : tensor<?x?xf32> into tensor<?x16xf32>
    %4 = affine.for %arg3 = 0 to 5 iter_args(%arg4 = %inserted_slice) -> (tensor<?x16xf32>) {
      %6 = affine.for %arg5 = 0 to 16 iter_args(%arg6 = %arg4) -> (tensor<?x16xf32>) {
        %7 = arith.index_cast %arg5 : index to i32
        %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
        %8 = arith.addi %7, %extracted : i32
        %9 = arith.cmpi sge, %8, %c0_i32 : i32
        %10 = arith.cmpi slt, %8, %c16_i32 : i32
        %11 = arith.andi %9, %10 : i1
        %12 = scf.if %11 -> (tensor<?x16xf32>) {
          %13 = arith.index_cast %8 : i32 to index
          %extracted_0 = tensor.extract %1[%arg3] : tensor<?xi32>
          %14 = arith.cmpi sge, %extracted_0, %c0_i32 : i32
          %15 = arith.subi %7, %extracted_0 : i32
          %16 = arith.select %14, %7, %15 : i32
          %17 = arith.index_cast %16 : i32 to index
          %extracted_1 = tensor.extract %2[%arg3, %17] : tensor<?x16xf32>
          %inserted = tensor.insert %extracted_1 into %arg6[%arg5, %13] : tensor<?x16xf32>
          scf.yield %inserted : tensor<?x16xf32>
        } else {
          scf.yield %arg6 : tensor<?x16xf32>
        }
        affine.yield %12 : tensor<?x16xf32>
      }
      affine.yield %6 : tensor<?x16xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?x16xf32>
    memref.copy %5, %arg2 : memref<?x16xf32> to memref<?x16xf32>
    return
  }
}

