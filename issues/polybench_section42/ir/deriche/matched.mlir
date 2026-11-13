#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant -2.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.negf %arg2 : f64
    %4 = math.exp %3 : f64
    %5 = arith.subf %cst_2, %4 : f64
    %6 = arith.mulf %5, %5 : f64
    %7 = arith.mulf %arg2, %cst_1 : f64
    %8 = arith.mulf %7, %4 : f64
    %9 = arith.addf %8, %cst_2 : f64
    %10 = math.exp %7 : f64
    %11 = arith.subf %9, %10 : f64
    %12 = arith.divf %6, %11 : f64
    %13 = arith.mulf %12, %4 : f64
    %14 = arith.subf %arg2, %cst_2 : f64
    %15 = arith.mulf %13, %14 : f64
    %16 = math.powf %cst_1, %3 : f64
    %17 = arith.mulf %arg2, %cst_0 : f64
    %18 = math.exp %17 : f64
    %19 = arith.negf %18 : f64
    %20 = arith.index_cast %arg0 : i32 to index
    %21 = tensor.empty(%20) : tensor<?xf64>
    %22 = tensor.empty(%20) : tensor<?xf64>
    %23 = tensor.empty(%20) : tensor<?xf64>
    %24 = kernel.launch @memset_zero_1D(%21) : (tensor<?xf64>) -> tensor<?xf64>
    %25 = kernel.launch @memset_zero_1D(%22) : (tensor<?xf64>) -> tensor<?xf64>
    %26 = kernel.launch @memset_zero_1D(%23) : (tensor<?xf64>) -> tensor<?xf64>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %0[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_4 = tensor.extract_slice %1[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_5 = tensor.extract_slice %26[0] [%20] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_6 = tensor.extract_slice %25[0] [%20] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_7 = tensor.extract_slice %24[0] [%20] [1] : tensor<?xf64> to tensor<?xf64>
    %27:4 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map2, #map2, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice, %extracted_slice_3 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%extracted_slice_4, %extracted_slice_5, %extracted_slice_6, %extracted_slice_7 : tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
    ^bb0(%in: f64, %in_8: f64, %out: f64, %out_9: f64, %out_10: f64, %out_11: f64):
      %29 = arith.mulf %12, %in : f64
      %30 = arith.mulf %15, %out_9 : f64
      %31 = arith.addf %29, %30 : f64
      %32 = arith.mulf %16, %out_11 : f64
      %33 = arith.addf %31, %32 : f64
      %34 = arith.mulf %19, %out_10 : f64
      %35 = arith.addf %33, %34 : f64
      linalg.yield %35, %in_8, %out_11, %35 : f64, f64, f64, f64
    } -> (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
    %inserted_slice = tensor.insert_slice %27#0 into %1[0, 0] [%20, %2] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %28 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %28, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

