#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_addmm_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x48xf32>, %arg4: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c48 = arith.constant 48 : index
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x48xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?x48xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?xi32>
    %cusparse_coo_rows_2215 = arith.constant 64 : index
    %cusparse_coo_nnz_2215 = arith.constant 512 : index
    %cusparse_coo_arg_2215_3 = memref.cast %arg3 : memref<?x48xf32> to memref<?x?xf32>
    %cusparse_coo_arg_2215_4 = memref.cast %arg4 : memref<?x48xf32> to memref<?x?xf32>
    kernel.launch @cusparseSpMM_COO_f32_memref(%cusparse_coo_rows_2215, %cusparse_coo_nnz_2215, %arg0, %arg1, %arg2, %cusparse_coo_arg_2215_3, %cusparse_coo_arg_2215_4) : (index, index, memref<?xi32>, memref<?xi32>, memref<?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}
