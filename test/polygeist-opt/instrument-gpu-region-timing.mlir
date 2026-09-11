// RUN: polygeist-opt '--instrument-gpu-region-timing=function=device_region' %s | FileCheck %s

module {
  func.func private @polygeist_cublas_test()

  func.func @device_region(%host: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %first = gpu.alloc () : memref<128xf32>
    %second = gpu.alloc () : memref<128xf32>
    gpu.memcpy %first, %host : memref<128xf32>, memref<128xf32>
    scf.for %iteration = %c0 to %c4 step %c1 {
      func.call @polygeist_cublas_test() : () -> ()
      gpu.memcpy %second, %first : memref<128xf32>, memref<128xf32>
      func.call @polygeist_cublas_test() : () -> ()
    }
    gpu.memcpy %host, %second : memref<128xf32>, memref<128xf32>
    gpu.dealloc %first : memref<128xf32>
    gpu.dealloc %second : memref<128xf32>
    return
  }

  func.func @untouched() {
    func.call @polygeist_cublas_test() : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @device_region
// CHECK-SAME: polygeist.gpu_timing_instrumented
// CHECK-SAME: polygeist.gpu_timing_region_id
// CHECK: %[[ID:.*]] = arith.constant {{.*}} : i64
// CHECK-NEXT: call @polygeist_gpu_region_timing_begin(%[[ID]])
// CHECK: %[[ALLOC0:.*]] = arith.constant 2 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[ALLOC0]])
// CHECK-NEXT: %[[FIRST:.*]] = gpu.alloc
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: %[[ALLOC1:.*]] = arith.constant 2 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[ALLOC1]])
// CHECK-NEXT: %[[SECOND:.*]] = gpu.alloc
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: %[[H2D:.*]] = arith.constant 4 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[H2D]])
// CHECK-NEXT: gpu.memcpy %[[FIRST]], %{{.*}}
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: %[[COMPUTE:.*]] = arith.constant 1 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[COMPUTE]])
// CHECK-NEXT: scf.for
// CHECK: func.call @polygeist_cublas_test()
// CHECK: %[[D2D:.*]] = arith.constant 6 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[D2D]])
// CHECK-NEXT: gpu.memcpy %[[SECOND]], %[[FIRST]]
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: func.call @polygeist_cublas_test()
// CHECK: }
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: %[[D2H:.*]] = arith.constant 5 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[D2H]])
// CHECK-NEXT: gpu.memcpy %{{.*}}, %[[SECOND]]
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: %[[FREE:.*]] = arith.constant 3 : i32
// CHECK-NEXT: call @polygeist_gpu_region_timing_enter(%[[FREE]])
// CHECK-NEXT: gpu.dealloc %[[FIRST]]
// CHECK-NEXT: call @polygeist_gpu_region_timing_leave()
// CHECK: call @polygeist_gpu_region_timing_end()
// CHECK-NEXT: return

// CHECK-LABEL: func.func @untouched
// CHECK-NOT: polygeist_gpu_region_timing
// CHECK: return
