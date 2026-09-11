#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 * 7)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_linear1d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.addf %9, %cst_0 : f32
        %11 = arith.mulf %10, %cst_1 : f32
        %12 = arith.divf %11, %cst_2 : f32
        %13 = arith.subf %12, %cst_0 : f32
        %14 = arith.cmpf olt, %13, %cst : f32
        %15 = arith.select %14, %cst, %13 : f32
        %16 = arith.fptosi %15 : f32 to i32
        %17 = arith.addi %6, %16 : i32
        %18 = arith.index_cast %17 : i32 to index
        %19 = affine.apply #map1(%arg4, %arg2)
        %extracted = tensor.extract %1[%19] : tensor<?xf32>
        %20 = arith.sitofp %16 : i32 to f32
        %21 = arith.subf %15, %20 : f32
        %22 = arith.subf %cst_3, %21 : f32
        %23 = arith.mulf %extracted, %22 : f32
        %extracted_4 = tensor.extract %arg5[%18] : tensor<?xf32>
        %24 = arith.addf %extracted_4, %23 : f32
        %inserted = tensor.insert %24 into %arg5[%18] : tensor<?xf32>
        %25 = arith.addi %16, %c1_i32 : i32
        %26 = arith.cmpi slt, %25, %c4_i32 : i32
        %27 = arith.select %26, %25, %16 : i32
        %28 = arith.addi %6, %27 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = affine.apply #map1(%arg4, %arg2)
        %extracted_5 = tensor.extract %1[%30] : tensor<?xf32>
        %31 = arith.mulf %extracted_5, %21 : f32
        %extracted_6 = tensor.extract %inserted[%29] : tensor<?xf32>
        %32 = arith.addf %extracted_6, %31 : f32
        %inserted_7 = tensor.insert %32 into %inserted[%29] : tensor<?xf32>
        affine.yield %inserted_7 : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

