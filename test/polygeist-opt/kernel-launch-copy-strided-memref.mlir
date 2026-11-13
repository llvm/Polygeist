// RUN: polygeist-opt --lower-kernel-launch-to-cublas --canonicalize %s | FileCheck %s
module {
  kernel.defn @cudaCopy1D_f32_memref(%x: memref<?xf32, strided<[?], offset: ?>>,
      %y: memref<?xf32, strided<[?], offset: ?>>) { kernel.yield }
  func.func @strided_column(%x: memref<?xf32, strided<[64], offset: 3>>,
      %y: memref<?xf32, strided<[2], offset: 5>>) {
    %xc = memref.cast %x : memref<?xf32, strided<[64], offset: 3>> to memref<?xf32, strided<[?], offset: ?>>
    %yc = memref.cast %y : memref<?xf32, strided<[2], offset: 5>> to memref<?xf32, strided<[?], offset: ?>>
    kernel.launch @cudaCopy1D_f32_memref(%xc, %yc) : (memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>) -> ()
    return
  }
}
// CHECK-LABEL: func.func @strided_column
// CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %[[SS:.*]] = memref.extract_strided_metadata
// CHECK: %{{.*}}, %{{.*}}, %{{.*}}, %[[DS:.*]] = memref.extract_strided_metadata
// CHECK: %[[S:.*]] = arith.index_cast %[[SS]] : index to i32
// CHECK: %[[D:.*]] = arith.index_cast %[[DS]] : index to i32
// CHECK: arith.muli
// CHECK: llvm.inttoptr
// CHECK: arith.muli
// CHECK: llvm.inttoptr
// CHECK: call @polygeist_cuda_copy_strided_2d_f32({{.*}}, %[[S]], {{.*}}, %[[D]],
// CHECK-NOT: kernel.launch
