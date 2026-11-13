// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s | FileCheck %s

// Aligned scratch storage must preserve alignment and coexist with ordinary
// allocations. Its 16-byte payload must be rounded up for aligned_alloc.
module {
  func.func private @observe(memref<4xi32>, memref<4xi32>)
  func.func @mixed_allocations() {
    %aligned = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
    %ordinary = memref.alloc() : memref<4xi32>
    call @observe(%aligned, %ordinary) : (memref<4xi32>, memref<4xi32>) -> ()
    memref.dealloc %aligned : memref<4xi32>
    memref.dealloc %ordinary : memref<4xi32>
    return
  }
}
// CHECK-LABEL: llvm.func @mixed_allocations()
// CHECK: llvm.call @aligned_alloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @observe
// CHECK: llvm.call @free
// CHECK: llvm.call @free
// CHECK: llvm.return

// -----

// Contiguous copies use their actual byte extent, including non-f32 elements.
module {
  func.func @copy_matrix(%src: memref<3x7xf64>, %dst: memref<3x7xf64>) {
    memref.copy %src, %dst : memref<3x7xf64> to memref<3x7xf64>
    return
  }
}
// CHECK-LABEL: llvm.func @copy_matrix
// CHECK: llvm.mlir.constant(21 : index)
// CHECK: llvm.intr.memcpy
// CHECK-NOT: memref.
// CHECK: llvm.return

// -----

// Array arguments can lose their static extent without changing their address.
module {
  func.func private @consume(memref<?xf32>)
  func.func @shape_cast(%src: memref<1xf32>) {
    %view = memref.cast %src : memref<1xf32> to memref<?xf32>
    call @consume(%view) : (memref<?xf32>) -> ()
    return
  }
}
// CHECK-LABEL: llvm.func @shape_cast
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr)
// CHECK-NOT: memref.
// CHECK: llvm.call @consume(%[[PTR]])
// CHECK: llvm.return

// -----

// Dynamic byte counts need the same alignment rounding.
module {
  func.func private @observe_dynamic(memref<?xi32>, index)
  func.func @dynamic_allocation(%n: index) {
    %buffer = memref.alloc(%n) {alignment = 64 : i64} : memref<?xi32>
    call @observe_dynamic(%buffer, %n) : (memref<?xi32>, index) -> ()
    memref.dealloc %buffer : memref<?xi32>
    return
  }
}
// CHECK-LABEL: llvm.func @dynamic_allocation
// CHECK: llvm.add
// CHECK: llvm.and
// CHECK: llvm.call @aligned_alloc
// CHECK: llvm.call @observe_dynamic
// CHECK: llvm.call @free
// CHECK: llvm.return

// -----

// Allocator declarations may already use LLVM pointer types.
module {
  func.func private @malloc(i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  func.func private @observe(memref<4xi32>)
  func.func @pointer_allocator() {
    %buffer = memref.alloc() : memref<4xi32>
    call @observe(%buffer) : (memref<4xi32>) -> ()
    memref.dealloc %buffer : memref<4xi32>
    return
  }
}
// CHECK-LABEL: llvm.func @pointer_allocator()
// CHECK: llvm.call @malloc
// CHECK: llvm.call @observe
// CHECK: llvm.call @free
// CHECK: llvm.return

// -----

// Source-level declarations of malloc/free can also use C-style memrefs.
module {
  func.func private @malloc(i64) -> memref<?xi8>
  func.func private @free(memref<?xi8>)
  func.func private @observe(memref<4xi32>)
  func.func @memref_allocator() {
    %buffer = memref.alloc() : memref<4xi32>
    call @observe(%buffer) : (memref<4xi32>) -> ()
    memref.dealloc %buffer : memref<4xi32>
    return
  }
}
// CHECK-LABEL: llvm.func @memref_allocator()
// CHECK: llvm.call @malloc
// CHECK: llvm.call @observe
// CHECK: llvm.call @free
// CHECK: llvm.return

// -----

// A cast to a less precise shape must not hide constant sizes and strides.
module {
  func.func @cast_metadata(%array: memref<3x7xf32>) -> (index, index, index) {
    %view = memref.cast %array : memref<3x7xf32> to memref<?x?xf32>
    %base, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %view
      : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
    return %sizes#0, %sizes#1, %strides#0 : index, index, index
  }
}
// CHECK-LABEL: llvm.func @cast_metadata
// CHECK-DAG: llvm.mlir.constant(3 : index)
// CHECK-DAG: llvm.mlir.constant(7 : index)
// CHECK-NOT: memref.
// CHECK: llvm.return
