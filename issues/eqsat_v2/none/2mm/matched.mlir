#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg7 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg8 : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg10 : memref<?x?xf64>
    %lv2_md1:6 = memref.extract_strided_metadata %arg8 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i2 = arith.index_cast %lv2_md1#4 : index to i32
    %lv2_md3:6 = memref.extract_strided_metadata %arg7 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i4 = arith.index_cast %lv2_md3#4 : index to i32
    %lv2_md5:6 = memref.extract_strided_metadata %arg6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i6 = arith.index_cast %lv2_md5#4 : index to i32
    %16 = kernel.launch @netlib_dgemm_nn_beta0(%arg1, %arg0, %arg2, %arg4, %lv2_i2, %2, %lv2_i4, %1, %lv2_i6, %0) : (i32, i32, i32, f64, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>) -> tensor<?x?xf64>
    %17 = bufferization.to_memref %16 : memref<?x?xf64>
    memref.copy %17, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    %lv2_c7 = arith.constant 1.0 : f64
    %lv2_md8:6 = memref.extract_strided_metadata %arg9 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i9 = arith.index_cast %lv2_md8#4 : index to i32
    %lv2_md10:6 = memref.extract_strided_metadata %arg6 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i11 = arith.index_cast %lv2_md10#4 : index to i32
    %lv2_md12:6 = memref.extract_strided_metadata %arg10 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i13 = arith.index_cast %lv2_md12#4 : index to i32
    %25 = kernel.launch @netlib_dgemm_nn(%arg3, %arg0, %arg1, %lv2_c7, %lv2_i9, %3, %lv2_i11, %16, %arg5, %lv2_i13, %4) : (i32, i32, i32, f64, i32, tensor<?x?xf64>, i32, tensor<?x?xf64>, f64, i32, tensor<?x?xf64>) -> tensor<?x?xf64>
    %26 = bufferization.to_memref %25 : memref<?x?xf64>
    memref.copy %26, %arg10 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

