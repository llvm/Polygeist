#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg7 : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg8 : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %5 = bufferization.to_tensor %arg10 : memref<?x?xf64>
    %6 = bufferization.to_tensor %arg11 : memref<?x?xf64>
    %lv2_c1 = arith.constant 1.0 : f64
    %lv2_md2:6 = memref.extract_strided_metadata %arg7 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i3 = arith.index_cast %lv2_md2#4 : index to i32
    %lv2_md4:6 = memref.extract_strided_metadata %arg6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i5 = arith.index_cast %lv2_md4#4 : index to i32
    %lv2_md6:6 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i7 = arith.index_cast %lv2_md6#4 : index to i32
    %19 = kernel.launch @netlib_dgemm_nn_beta0(%arg1, %arg0, %arg2, %lv2_c1, %lv2_i3, %2, %lv2_i5, %1, %lv2_i7, %0) : (i32, i32, i32, f64, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>) -> tensor<?x?xf64>
    %20 = bufferization.to_memref %19 : memref<?x?xf64>
    memref.copy %20, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %lv2_c8 = arith.constant 1.0 : f64
    %lv2_md9:6 = memref.extract_strided_metadata %arg10 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i10 = arith.index_cast %lv2_md9#4 : index to i32
    %lv2_md11:6 = memref.extract_strided_metadata %arg9 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i12 = arith.index_cast %lv2_md11#4 : index to i32
    %lv2_md13:6 = memref.extract_strided_metadata %arg8 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i14 = arith.index_cast %lv2_md13#4 : index to i32
    %28 = kernel.launch @netlib_dgemm_nn_beta0(%arg3, %arg1, %arg4, %lv2_c8, %lv2_i10, %5, %lv2_i12, %4, %lv2_i14, %3) : (i32, i32, i32, f64, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>) -> tensor<?x?xf64>
    %29 = bufferization.to_memref %28 : memref<?x?xf64>
    memref.copy %29, %arg8 : memref<?x?xf64> to memref<?x?xf64>
    %lv2_c15 = arith.constant 1.0 : f64
    %lv2_md16:6 = memref.extract_strided_metadata %arg8 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i17 = arith.index_cast %lv2_md16#4 : index to i32
    %lv2_md18:6 = memref.extract_strided_metadata %arg5 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i19 = arith.index_cast %lv2_md18#4 : index to i32
    %lv2_md20:6 = memref.extract_strided_metadata %arg11 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i21 = arith.index_cast %lv2_md20#4 : index to i32
    %37 = kernel.launch @netlib_dgemm_nn_beta0(%arg3, %arg0, %arg1, %lv2_c15, %lv2_i17, %28, %lv2_i19, %19, %lv2_i21, %6) : (i32, i32, i32, f64, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>) -> tensor<?x?xf64>
    %38 = bufferization.to_memref %37 : memref<?x?xf64>
    memref.copy %38, %arg11 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

