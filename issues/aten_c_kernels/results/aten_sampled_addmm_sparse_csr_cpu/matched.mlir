#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sampled_addmm_sparse_csr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x32xf32>, %arg4: memref<?x24xf32>, %arg5: f32, %arg6: f32, %arg7: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg7 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xi32>
    %3 = bufferization.to_tensor %arg0 : memref<?xi32>
    %cusparse_sddmm_rows_1802 = arith.constant 16 : index
    %cusparse_sddmm_a_1802 = memref.cast %arg3 : memref<?x32xf32> to memref<?x?xf32>
    %cusparse_sddmm_b_1802 = memref.cast %arg4 : memref<?x24xf32> to memref<?x?xf32>
    kernel.launch @cusparseSDDMM_CSR_f32_memref(%cusparse_sddmm_rows_1802, %arg0, %arg1, %arg2, %cusparse_sddmm_a_1802, %cusparse_sddmm_b_1802, %arg5, %arg6, %arg7) : (index, memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?x?xf32>, memref<?x?xf32>, f32, f32, memref<?xf32>) -> ()
    return
  }
}

