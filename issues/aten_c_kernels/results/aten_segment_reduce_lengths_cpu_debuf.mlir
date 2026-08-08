#map = affine_map<()[s0] -> (s0 - 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_segment_reduce_lengths_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1 = arith.constant -1 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = arith.index_cast %arg2 : i32 to index
    %4 = arith.cmpi eq, %arg2, %c2_i32 : i32
    %5 = arith.cmpi eq, %arg2, %c0_i32 : i32
    %6 = arith.cmpi eq, %arg2, %c3_i32 : i32
    %7 = arith.select %6, %cst_0, %cst : f32
    %8 = arith.select %4, %cst_1, %7 : f32
    %9 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %10 = arith.select %5, %true, %9 : i1
    %11 = arith.addi %3, %c-1 : index
    %12 = arith.cmpi eq, %11, %c0 : index
    %13 = tensor.empty() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %13[] : tensor<i32>
    %14:2 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %inserted, %arg6 = %0) -> (tensor<i32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg5[] : tensor<i32>
      %extracted_2 = tensor.extract %1[%arg4] : tensor<?xi32>
      %16 = arith.index_cast %extracted_2 : i32 to index
      %17 = arith.index_cast %extracted : i32 to index
      %18 = arith.addi %17, %16 : index
      %19 = arith.index_cast %18 : index to i32
      %20 = scf.for %arg7 = %c0 to %16 step %c1 iter_args(%arg8 = %8) -> (f32) {
        %26 = arith.addi %17, %arg7 : index
        %extracted_5 = tensor.extract %2[%26] : tensor<?xf32>
        %27 = scf.if %10 -> (f32) {
          %28 = arith.addf %arg8, %extracted_5 : f32
          scf.yield %28 : f32
        } else {
          %28 = affine.apply #map()[%3]
          %29 = arith.cmpi eq, %28, %c0 : index
          %30 = arith.cmpf ogt, %arg8, %extracted_5 : f32
          %31 = arith.select %30, %arg8, %extracted_5 : f32
          %32 = arith.cmpf olt, %arg8, %extracted_5 : f32
          %33 = arith.select %32, %arg8, %extracted_5 : f32
          %34 = arith.select %29, %31, %33 : f32
          scf.yield %34 : f32
        }
        scf.yield %27 : f32
      }
      %21 = arith.cmpi ne, %extracted_2, %c0_i32 : i32
      %22 = arith.andi %12, %21 : i1
      %23 = arith.sitofp %extracted_2 : i32 to f32
      %24 = arith.divf %20, %23 : f32
      %25 = arith.select %22, %24, %20 : f32
      %inserted_3 = tensor.insert %25 into %arg6[%arg4] : tensor<?xf32>
      %inserted_4 = tensor.insert %19 into %arg5[] : tensor<i32>
      affine.yield %inserted_4, %inserted_3 : tensor<i32>, tensor<?xf32>
    }
    %15 = bufferization.to_memref %14#1 : memref<?xf32>
    memref.copy %15, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

