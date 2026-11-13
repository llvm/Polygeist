// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasDgemm_simple(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  func.func @split_k(%a: memref<4x16xf64>, %b: memref<16x6xf64>,
                     %c: memref<4x6xf64>) {
    %at = bufferization.to_tensor %a restrict : memref<4x16xf64>
    %bt = bufferization.to_tensor %b restrict : memref<16x6xf64>
    %ct = bufferization.to_tensor %c restrict writable : memref<4x6xf64>
    %a0 = tensor.extract_slice %at[0, 0] [4, 8] [1, 1]
      : tensor<4x16xf64> to tensor<4x8xf64>
    %b0 = tensor.extract_slice %bt[0, 0] [8, 6] [1, 1]
      : tensor<16x6xf64> to tensor<8x6xf64>
    %a0d = tensor.cast %a0 : tensor<4x8xf64> to tensor<?x?xf64>
    %b0d = tensor.cast %b0 : tensor<8x6xf64> to tensor<?x?xf64>
    %ctd = tensor.cast %ct : tensor<4x6xf64> to tensor<?x?xf64>
    %r0 = kernel.launch @cublasDgemm_simple(%a0d, %b0d, %ctd)
      : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
        -> tensor<?x?xf64>
    %a1 = tensor.extract_slice %at[0, 8] [4, 8] [1, 1]
      : tensor<4x16xf64> to tensor<4x8xf64>
    %b1 = tensor.extract_slice %bt[8, 0] [8, 6] [1, 1]
      : tensor<16x6xf64> to tensor<8x6xf64>
    %a1d = tensor.cast %a1 : tensor<4x8xf64> to tensor<?x?xf64>
    %b1d = tensor.cast %b1 : tensor<8x6xf64> to tensor<?x?xf64>
    %r1 = kernel.launch @cublasDgemm_simple(%a1d, %b1d, %r0)
      : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
        -> tensor<?x?xf64>
    %r1s = tensor.cast %r1 : tensor<?x?xf64> to tensor<4x6xf64>
    %out = bufferization.to_memref %r1s : memref<4x6xf64>
    memref.copy %out, %c : memref<4x6xf64> to memref<4x6xf64>
    return
  }

  func.func @transposed_a(%a: tensor<?x?xf64>, %b: tensor<?x?xf64>,
                          %c: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %r = kernel.launch @cublasDgemm_simple(%a, %b, %c) {
      polygeist.gemm_trans_a = true,
      polygeist.gemm_trans_b = false
    } : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
      -> tensor<?x?xf64>
    return %r : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @split_k
// CHECK: %[[A0:.*]] = memref.subview %arg0[0, 0] [4, 8] [1, 1]
// CHECK: %[[B0:.*]] = memref.subview %arg1[0, 0] [8, 6] [1, 1]
// CHECK: memref.extract_strided_metadata %[[A0]] : memref<4x8xf64, strided<[16, 1]>>
// CHECK: %[[ALDA:.*]] = arith.index_cast %{{.*}}#0 : index to i32
// CHECK: memref.extract_strided_metadata %[[B0]] : memref<8x6xf64, strided<[6, 1]>>
// CHECK: %[[BLDA:.*]] = arith.index_cast %{{.*}}#0 : index to i32
// CHECK: call @polygeist_cublas_dgemm_transpose
// CHECK-SAME: %[[ALDA]]
// CHECK-SAME: %[[BLDA]]
// CHECK: %[[A1:.*]] = memref.subview %arg0[0, 8] [4, 8] [1, 1] : {{.*}}offset: 8
// CHECK: %[[B1:.*]] = memref.subview %arg1[8, 0] [8, 6] [1, 1] : {{.*}}offset: 48
// CHECK: memref.extract_strided_metadata %[[A1]]
// CHECK: memref.extract_aligned_pointer_as_index %[[A1]]
// CHECK: %{{.*}}, %[[AOFF:.*]], %{{.*}}:2, %{{.*}}:2 = memref.extract_strided_metadata %[[A1]]
// CHECK: %[[AOFF64:.*]] = arith.index_cast %[[AOFF]] : index to i64
// CHECK: %[[ABYTES:.*]] = arith.muli %[[AOFF64]], %{{.*}} : i64
// CHECK: %[[APTR:.*]] = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: call @polygeist_cublas_dgemm_transpose
// CHECK-SAME: %[[APTR]]
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @transposed_a
// CHECK: %[[TA:.*]] = arith.constant 1 : i32
// CHECK: %[[TB:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cublas_dgemm_transpose
// CHECK-SAME: %[[TA]], %[[TB]]
// CHECK-NOT: kernel.launch
