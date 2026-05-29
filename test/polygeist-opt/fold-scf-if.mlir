// RUN: polygeist-opt --fold-scf-if --split-input-file %s | FileCheck %s

func.func @store_select(%A: memref<10xf32>, %a: f32, %b: f32, %cond: i1) {
  scf.if %cond {
    affine.store %a, %A[0] : memref<10xf32>
  } else {
    affine.store %b, %A[0] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func.func @store_select
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : f32
// CHECK: affine.store %[[SELECT]], %{{.*}}[0] : memref<10xf32>
// CHECK: return

// -----

func.func @guarded_load(%A: memref<?xf32>, %B: memref<?xf32>, %i: index,
                        %cond: i1) {
  scf.if %cond {
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %B[%i] : memref<?xf32>
  } else {
    %z = arith.constant 0.000000e+00 : f32
    memref.store %z, %B[%i] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: func.func @guarded_load
// CHECK: scf.if
// CHECK: memref.load
// CHECK: memref.store
// CHECK: return
