// RUN: polygeist-opt '--plan-gpu-data-residency=function=session promote-function-arguments=false' %s | FileCheck %s
// RUN: polygeist-opt '--plan-gpu-data-residency=function=session promote-function-arguments=false' %s | mlir-opt --expand-strided-metadata --lower-affine --convert-scf-to-cf --gpu-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

module {
  func.func private @polygeist_cublas_test(index)
  func.func @session() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %scratch = memref.alloc() : memref<128xf32>
      %ptr = memref.extract_aligned_pointer_as_index %scratch : memref<128xf32> -> index
      func.call @polygeist_cublas_test(%ptr) : (index) -> ()
      memref.dealloc %scratch : memref<128xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @session
// CHECK: %[[BUFFER:.*]], %{{.*}} = gpu.alloc async
// CHECK: scf.for
// CHECK-NOT: gpu.alloc
// CHECK: call @polygeist_cublas_test
// CHECK-NOT: gpu.dealloc
// CHECK: }
// CHECK: %[[STREAM:.*]] = gpu.wait async
// CHECK: %[[DONE:.*]] = gpu.dealloc async [%[[STREAM]]] %[[BUFFER]]
// CHECK: gpu.wait [%[[DONE]]]
// CHECK-NEXT: return

// LLVM-LABEL: llvm.func @session
// LLVM: llvm.call @mgpuMemAlloc
// LLVM: llvm.call @polygeist_cublas_test
// LLVM: llvm.call @mgpuMemFree
// LLVM: llvm.return
