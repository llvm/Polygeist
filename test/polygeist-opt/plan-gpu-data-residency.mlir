// RUN: polygeist-opt '--plan-gpu-data-residency=function=device_region' %s | FileCheck %s --check-prefix=RESIDENT
// RUN: polygeist-opt '--plan-gpu-data-residency=function=host_interleaved' %s | FileCheck %s --check-prefix=FALLBACK
// RUN: polygeist-opt '--plan-gpu-data-residency=function=library_region' %s | FileCheck %s --check-prefix=LIBRARY
// RUN: polygeist-opt '--plan-gpu-data-residency=function=possibly_aliasing' %s | FileCheck %s --check-prefix=ALIAS-FALLBACK
// RUN: polygeist-opt '--plan-gpu-data-residency=function=repeated_owner' %s | FileCheck %s --check-prefix=CALLEE
// RUN: polygeist-opt '--plan-gpu-data-residency=function=local_scratch promote-function-arguments=false' %s | FileCheck %s --check-prefix=SCRATCH

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @touch(%arg0: memref<?xf32>) kernel {
      gpu.return
    }
    gpu.func @touch_static(%arg0: memref<128xf32>) kernel {
      gpu.return
    }
  }

  func.func @device_region(%buffer: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %unranked = memref.cast %buffer : memref<?xf32> to memref<*xf32>
    gpu.host_register %unranked : memref<*xf32>
    scf.for %iteration = %c0 to %c4 step %c1 {
      gpu.launch_func @kernels::@touch
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%buffer : memref<?xf32>)
    }
    return
  }

  func.func @host_interleaved(%buffer: memref<?xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@touch
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%buffer : memref<?xf32>)
    %value = memref.load %buffer[%c0] : memref<?xf32>
    return %value : f32
  }

  func.func private @polygeist_cublas_test(!llvm.ptr)

  func.func @library_region(%buffer: memref<128xf32>) {
    %address = memref.extract_aligned_pointer_as_index %buffer : memref<128xf32> -> index
    %integer = arith.index_cast %address : index to i64
    %pointer = llvm.inttoptr %integer : i64 to !llvm.ptr
    call @polygeist_cublas_test(%pointer) : (!llvm.ptr) -> ()
    return
  }

  func.func @possibly_aliasing(%first: memref<128xf32>,
                               %second: memref<128xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@touch_static
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%first : memref<128xf32>)
    gpu.launch_func @kernels::@touch_static
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%second : memref<128xf32>)
    return
  }

  func.func @owned_helper(%buffer: memref<?xf32>) attributes {
      polygeist.gpu_resident_callee} {
    %c1 = arith.constant 1 : index
    %unranked = memref.cast %buffer : memref<?xf32> to memref<*xf32>
    gpu.host_register %unranked : memref<*xf32>
    gpu.launch_func @kernels::@touch
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%buffer : memref<?xf32>)
    return
  }

  func.func @repeated_owner(%count: index, %buffer: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %count step %c1 {
      func.call @owned_helper(%buffer) : (memref<?xf32>) -> ()
    }
    return
  }

  func.func @local_scratch(%count: index, %shape: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %count step %c1 {
      %size = memref.dim %shape, %c0 : memref<?xf32>
      %scratch = memref.alloc(%size) : memref<?xf32>
      gpu.launch_func @kernels::@touch
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%scratch : memref<?xf32>)
    }
    return
  }
}

// RESIDENT-LABEL: func.func @device_region
// RESIDENT-SAME: attributes {polygeist.gpu_data_residency}
// RESIDENT: %[[DEVICE:.*]], %[[ALLOC_TOKEN:.*]] = gpu.alloc async
// RESIDENT: gpu.memcpy async [%[[ALLOC_TOKEN]]] %[[DEVICE]], %[[HOST:[a-zA-Z0-9_]+]]
// RESIDENT-NOT: gpu.host_register
// RESIDENT: scf.for
// RESIDENT: gpu.launch_func
// RESIDENT-SAME: args(%[[DEVICE]] : memref<?xf32>)
// RESIDENT: }
// RESIDENT: gpu.memcpy async {{.*}} %[[HOST]], %[[DEVICE]]
// RESIDENT: gpu.dealloc async {{.*}} %[[DEVICE]] : memref<?xf32>
// RESIDENT: return

// FALLBACK-LABEL: func.func @host_interleaved
// FALLBACK-NOT: gpu.alloc
// FALLBACK: gpu.launch_func
// FALLBACK-SAME: args(%[[HOST:.*]] : memref<?xf32>)
// FALLBACK: memref.load %[[HOST]]

// LIBRARY-LABEL: func.func @library_region
// LIBRARY: %[[DEVICE:.*]], %[[ALLOC_TOKEN:.*]] = gpu.alloc async
// LIBRARY: gpu.memcpy async [%[[ALLOC_TOKEN]]] %[[DEVICE]], %[[HOST:[a-zA-Z0-9_]+]]
// LIBRARY: memref.extract_aligned_pointer_as_index %[[DEVICE]]
// LIBRARY: call @polygeist_cublas_test
// LIBRARY: gpu.memcpy async {{.*}} %[[HOST]], %[[DEVICE]]
// LIBRARY: gpu.dealloc async {{.*}} %[[DEVICE]] : memref<128xf32>

// ALIAS-FALLBACK-LABEL: func.func @possibly_aliasing
// ALIAS-FALLBACK-NOT: gpu.alloc
// ALIAS-FALLBACK: gpu.launch_func
// ALIAS-FALLBACK: gpu.launch_func

// CALLEE-LABEL: func.func @owned_helper
// CALLEE-NOT: gpu.host_register
// CALLEE: gpu.launch_func
// CALLEE-LABEL: func.func @repeated_owner
// CALLEE-SAME: attributes {polygeist.gpu_data_residency}
// CALLEE: %[[DEVICE:.*]], %[[TOKEN:.*]] = gpu.alloc async
// CALLEE: gpu.memcpy async [%[[TOKEN]]] %[[DEVICE]], %[[HOST:[a-zA-Z0-9_]+]]
// CALLEE: scf.for
// CALLEE: func.call @owned_helper(%[[DEVICE]])
// CALLEE: gpu.memcpy async {{.*}} %[[HOST]], %[[DEVICE]]
// CALLEE: gpu.dealloc async {{.*}} %[[DEVICE]]

// SCRATCH-LABEL: func.func @local_scratch
// SCRATCH: %[[SIZE:.*]] = memref.dim %[[SHAPE:.*]], %{{.*}} : memref<?xf32>
// SCRATCH: %[[STREAM:.*]] = gpu.wait async
// SCRATCH: %[[DEVICE:.*]], %[[TOKEN:.*]] = gpu.alloc async [%[[STREAM]]] (%[[SIZE]])
// SCRATCH: gpu.wait [%[[TOKEN]]]
// SCRATCH: scf.for
// SCRATCH-NOT: memref.alloc
// SCRATCH: gpu.launch_func
// SCRATCH-SAME: args(%[[DEVICE]] : memref<?xf32>)
