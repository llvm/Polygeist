// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s
module {
  kernel.defn @cublasJointMaxAbsProduct_f32_memref(
      %a: memref<?xf32>, %b: memref<?xf32>, %out: memref<?xf32>) {
    kernel.yield
  }
  func.func @joint(%a: memref<?xf32>, %b: memref<?xf32>,
                   %out: memref<?xf32>) {
    kernel.launch @cublasJointMaxAbsProduct_f32_memref(%a, %b, %out) :
        (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cublas_joint_maxabs_product_f32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
