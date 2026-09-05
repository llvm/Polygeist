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
// CHECK: %[[VALUE:.*]] = memref.load
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %[[VALUE]], %[[ZERO]]
// CHECK: memref.store %[[SELECT]]
// CHECK-NOT: scf.if
// CHECK: return

// -----

func.func @guarded_max_store(%A: memref<?xf32>, %max: memref<f32>,
                             %i: index) {
  %candidate = affine.load %A[%i] : memref<?xf32>
  %old = affine.load %max[] : memref<f32>
  %cmp = arith.cmpf ogt, %candidate, %old : f32
  scf.if %cmp {
    %candidate_reload = affine.load %A[%i] : memref<?xf32>
    affine.store %candidate_reload, %max[] : memref<f32>
  }
  return
}

// CHECK-LABEL: func.func @guarded_max_store
// CHECK: %[[CANDIDATE:.*]] = affine.load %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK: %[[OLD:.*]] = affine.load %{{.*}}[] : memref<f32>
// CHECK: %[[CMP:.*]] = arith.cmpf ogt, %[[CANDIDATE]], %[[OLD]] : f32
// CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[CANDIDATE]], %[[OLD]] : f32
// CHECK: affine.store %[[SELECT]], %{{.*}}[] : memref<f32>
// CHECK-NOT: scf.if
// CHECK: return

// -----

memref.global @handle : memref<1xmemref<?xi32>> = uninitialized

func.func @branch_local_store_target(%value: memref<?xi32>, %cond: i1) {
  scf.if %cond {
    %handle = memref.get_global @handle : memref<1xmemref<?xi32>>
    affine.store %value, %handle[0] : memref<1xmemref<?xi32>>
  }
  return
}

// CHECK-LABEL: func.func @branch_local_store_target
// CHECK: %[[HANDLE:.*]] = memref.get_global @handle
// CHECK: %[[OLD:.*]] = affine.load %[[HANDLE]][0]
// CHECK: %[[SELECT:.*]] = arith.select %{{.*}}, %{{.*}}, %[[OLD]]
// CHECK: affine.store %[[SELECT]], %[[HANDLE]][0]
// CHECK-NOT: scf.if
// CHECK: return
