// RUN: polygeist-opt '--wrap-kernel-launch-pipeline=cuda-graphs=true capture-host-mapped-libraries=true maximal-device-sequence=true coalesce-control-flow=true' %s | FileCheck %s
module {
  func.func private @polygeist_cuda_copy_f32(i32, !llvm.ptr, !llvm.ptr)
  func.func @copies(%input: memref<128xf32>, %output: memref<128xf32>) {
    %n = arith.constant 128 : i32
    %a = memref.extract_aligned_pointer_as_index %input : memref<128xf32> -> index
    %ai = arith.index_cast %a : index to i64
    %ap = llvm.inttoptr %ai : i64 to !llvm.ptr
    %b = memref.extract_aligned_pointer_as_index %output : memref<128xf32> -> index
    %bi = arith.index_cast %b : index to i64
    %bp = llvm.inttoptr %bi : i64 to !llvm.ptr
    call @polygeist_cuda_copy_f32(%n, %ap, %bp) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    %a2 = memref.extract_aligned_pointer_as_index %input : memref<128xf32> -> index
    %a2i = arith.index_cast %a2 : index to i64
    %a2p = llvm.inttoptr %a2i : i64 to !llvm.ptr
    call @polygeist_cuda_copy_f32(%n, %bp, %a2p) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}
// CHECK-LABEL: func.func @copies
// CHECK: call @polygeist_cuda_graph_begin
// CHECK: call @polygeist_cublas_pipeline_begin
// CHECK: scf.if
// CHECK: call @polygeist_cuda_copy_f32
// CHECK-NOT: call @polygeist_cublas_pipeline_end
// CHECK-NOT: call @polygeist_cuda_graph_end
// CHECK: call @polygeist_cuda_copy_f32
// CHECK: call @polygeist_cuda_graph_end
// CHECK: call @polygeist_cublas_pipeline_end
// CHECK-NOT: call @polygeist_cuda_graph_begin
// CHECK: return
