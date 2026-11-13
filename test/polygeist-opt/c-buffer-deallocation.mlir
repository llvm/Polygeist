// RUN: polygeist-opt --buffer-deallocation --convert-polygeist-to-llvm %s | FileCheck %s

// Whole C programs contain existing frees as well as bufferization temporaries.
// The upstream allocation-based pass must free the temporary and preserve the
// explicit free, with one free per allocation.
module {
  func.func private @observe(memref<4xf32>, memref<4xf32>)
  func.func @temporary_and_explicit_free() {
    %temporary = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    %explicit = memref.alloc() : memref<4xf32>
    call @observe(%temporary, %explicit) : (memref<4xf32>, memref<4xf32>) -> ()
    memref.dealloc %explicit : memref<4xf32>
    return
  }
}
// CHECK-LABEL: llvm.func @temporary_and_explicit_free
// CHECK: llvm.call @aligned_alloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @observe
// CHECK-COUNT-2: llvm.call @free
// CHECK-NOT: llvm.call @free
// CHECK: llvm.return
