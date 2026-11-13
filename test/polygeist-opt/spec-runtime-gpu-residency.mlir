// RUN: polygeist-opt '--plan-gpu-data-residency=function=region' '--instrument-gpu-region-timing=function=region' %s | FileCheck %s
// RUN: polygeist-opt '--plan-gpu-data-residency=function=unknown' %s | FileCheck %s --check-prefix=UNKNOWN
// RUN: polygeist-opt '--plan-gpu-data-residency=function=metadata_dimensions' %s | FileCheck %s --check-prefix=METADATA
// RUN: polygeist-opt '--plan-gpu-data-residency=function=stack_scratch' %s | FileCheck %s --check-prefix=STACK
// RUN: polygeist-opt '--plan-gpu-data-residency=function=stack_host_read' %s | FileCheck %s --check-prefix=STACK-HOST

module {
  // This ABI contract permits device pointers; it does not promise graph safety.
  func.func private @imported_runtime(!llvm.ptr) attributes {polygeist.gpu_runtime}
  func.func private @unmarked_runtime(!llvm.ptr)
  func.func @region(%buffer: memref<128xf64>) {
    %address = memref.extract_aligned_pointer_as_index %buffer : memref<128xf64> -> index
    %integer = arith.index_cast %address : index to i64
    %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
    call @imported_runtime(%pointer) : (!llvm.ptr) -> ()
    return
  }
  func.func @unknown(%buffer: memref<128xf64>) {
    %address = memref.extract_aligned_pointer_as_index %buffer : memref<128xf64> -> index
    %integer = arith.index_cast %address : index to i64
    %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
    call @unmarked_runtime(%pointer) : (!llvm.ptr) -> ()
    return
  }
  func.func @metadata_dimensions(%buffer: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %size = memref.dim %buffer, %c0 : memref<?xf64>
    %scratch = memref.alloc(%size) : memref<?xf64>
    %address = memref.extract_aligned_pointer_as_index %buffer : memref<?xf64> -> index
    %integer = arith.index_cast %address : index to i64
    %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
    call @imported_runtime(%pointer) : (!llvm.ptr) -> ()
    %scratch_address = memref.extract_aligned_pointer_as_index %scratch : memref<?xf64> -> index
    %scratch_integer = arith.index_cast %scratch_address : index to i64
    %scratch_pointer = llvm.inttoptr %scratch_integer : i64 to !llvm.ptr
    call @imported_runtime(%scratch_pointer) : (!llvm.ptr) -> ()
    memref.dealloc %scratch : memref<?xf64>
    return
  }
  func.func @stack_scratch() {
    %scratch = memref.alloca() : memref<f32>
    %address = memref.extract_aligned_pointer_as_index %scratch : memref<f32> -> index
    %integer = arith.index_cast %address : index to i64
    %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
    call @imported_runtime(%pointer) : (!llvm.ptr) -> ()
    return
  }
  func.func @stack_host_read() -> f32 {
    %scratch = memref.alloca() : memref<f32>
    %address = memref.extract_aligned_pointer_as_index %scratch : memref<f32> -> index
    %integer = arith.index_cast %address : index to i64
    %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
    call @imported_runtime(%pointer) : (!llvm.ptr) -> ()
    %result = memref.load %scratch[] : memref<f32>
    return %result : f32
  }
}
// CHECK-LABEL: func.func @region
// CHECK-SAME: polygeist.gpu_timing_instrumented
// CHECK: %[[DEVICE:[^, ]+]], %{{.*}} = gpu.alloc
// CHECK: gpu.memcpy {{.*}} %[[DEVICE]], %{{.*}}
// CHECK: memref.extract_aligned_pointer_as_index %[[DEVICE]]
// CHECK: %[[COMPUTE:.*]] = arith.constant 1 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[COMPUTE]])
// CHECK-NEXT: call @imported_runtime
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: gpu.memcpy {{.*}} %{{.*}}, %[[DEVICE]]
// CHECK: return
// UNKNOWN-LABEL: func.func @unknown
// UNKNOWN-NOT: gpu.alloc
// UNKNOWN-NOT: gpu.memcpy
// UNKNOWN: call @unmarked_runtime
// UNKNOWN: return
// METADATA-LABEL: func.func @metadata_dimensions
// METADATA-SAME: polygeist.gpu_data_residency
// METADATA-NOT: memref.alloc
// METADATA: gpu.alloc
// METADATA: gpu.memcpy
// METADATA-NOT: memref.alloc
// METADATA: gpu.alloc
// METADATA: call @imported_runtime
// METADATA: call @imported_runtime
// METADATA: return
// STACK-LABEL: func.func @stack_scratch
// STACK-NOT: memref.alloca
// STACK: gpu.alloc
// STACK: call @imported_runtime
// STACK: gpu.dealloc
// STACK: return
// STACK-HOST-LABEL: func.func @stack_host_read
// STACK-HOST-NOT: gpu.alloc
// STACK-HOST: memref.alloca
// STACK-HOST: memref.load
// STACK-HOST: return
