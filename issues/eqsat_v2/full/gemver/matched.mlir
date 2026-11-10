#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = bufferization.to_tensor %arg5 : memref<?xf64>
    %3 = bufferization.to_tensor %arg6 : memref<?xf64>
    %4 = bufferization.to_tensor %arg7 : memref<?xf64>
    %5 = bufferization.to_tensor %arg8 : memref<?xf64>
    %6 = bufferization.to_tensor %arg9 : memref<?xf64>
    %7 = bufferization.to_tensor %arg10 : memref<?xf64>
    %8 = bufferization.to_tensor %arg11 : memref<?xf64>
    %9 = arith.index_cast %arg0 : i32 to index
    %10 = polygeist.submap(%1, %9, %9) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %11 = polygeist.submap(%2, %9, %9) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %12 = polygeist.submap(%3, %9, %9) {map = #map} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %13 = polygeist.submap(%4, %9, %9) {map = #map1} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %14 = polygeist.submap(%0, %9, %9) {map = #map2} : (tensor<?x?xf64>, index, index) -> tensor<?x?xf64>
    %rm15_18_c1 = arith.constant 1.0 : f64
    %rm15_18_md2:6 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %rm15_18_ld3 = arith.index_cast %rm15_18_md2#4 : index to i32
    kernel.launch @netlib_dger(%arg0, %arg0, %rm15_18_c1, %arg5, %arg4, %rm15_18_ld3, %arg3) : (i32, i32, f64, memref<?xf64>, memref<?xf64>, i32, memref<?x?xf64>) -> ()
    kernel.launch @netlib_dger(%arg0, %arg0, %rm15_18_c1, %arg7, %arg6, %rm15_18_ld3, %arg3) : (i32, i32, f64, memref<?xf64>, memref<?xf64>, i32, memref<?x?xf64>) -> ()
    %16 = bufferization.to_tensor %arg3 : memref<?x?xf64>
    %lv2_md1:6 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i2 = arith.index_cast %lv2_md1#4 : index to i32
    %22 = kernel.launch @netlib_dgemv_n_beta1(%arg0, %arg0, %arg2, %lv2_i2, %16, %7, %6) : (i32, i32, f64, i32, tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %lv2_c5 = arith.constant 1.0 : f64
    %lv2_c6 = arith.constant 1 : i32
    %lv2_c7 = arith.constant 1 : i32
    %26 = kernel.launch @netlib_daxpy_inc(%arg0, %lv2_c5, %8, %lv2_c6, %22, %lv2_c7) : (i32, f64, tensor<?xf64>, i32, tensor<?xf64>, i32) -> tensor<?xf64>
    %27 = bufferization.to_memref %26 : memref<?xf64>
    memref.copy %27, %arg9 : memref<?xf64> to memref<?xf64>
    %lv2_md3:6 = memref.extract_strided_metadata %arg3 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i4 = arith.index_cast %lv2_md3#4 : index to i32
    %32 = kernel.launch @netlib_dgemv_t_beta1(%arg0, %arg0, %arg1, %lv2_i4, %16, %26, %5) : (i32, i32, f64, i32, tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %33 = bufferization.to_memref %32 : memref<?xf64>
    memref.copy %33, %arg8 : memref<?xf64> to memref<?xf64>
    return
  }
}

