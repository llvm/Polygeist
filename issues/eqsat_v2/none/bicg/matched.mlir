#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_bicg(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg5 : memref<?xf64>
    %4 = bufferization.to_tensor %arg6 : memref<?xf64>
    %lv2_c1 = arith.constant 1.0 : f64
    %lv2_md2:6 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i3 = arith.index_cast %lv2_md2#4 : index to i32
    %17 = kernel.launch @netlib_dgemv_n_beta0(%arg0, %arg1, %lv2_c1, %lv2_i3, %0, %4, %1) : (i32, i32, f64, i32, tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %18 = bufferization.to_memref %17 : memref<?xf64>
    memref.copy %18, %arg3 : memref<?xf64> to memref<?xf64>
    %lv2_c4 = arith.constant 1.0 : f64
    %lv2_md5:6 = memref.extract_strided_metadata %arg2 : memref<?x?xf64> -> memref<f64>, index, index, index, index, index
    %lv2_i6 = arith.index_cast %lv2_md5#4 : index to i32
    %23 = kernel.launch @netlib_dgemv_t_beta0(%arg0, %arg1, %lv2_c4, %lv2_i6, %0, %3, %2) : (i32, i32, f64, i32, tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %24 = bufferization.to_memref %23 : memref<?xf64>
    memref.copy %24, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}

