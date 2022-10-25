// RUN: polygeist-opt --parallel-licm --split-input-file %s | FileCheck %s

module {
  func.func private @use(f32)
  func.func @hoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %a = memref.alloca() : memref<f32>
    memref.store %cst, %a[] : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
  func.func @hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %a = memref.alloca() : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      memref.store %cst, %a[] : memref<f32>
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
  func.func @hoist3(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %a : memref<f32>) {
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
  func.func private @get() -> (f32)
  func.func @nohoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %c1 = arith.constant 1 : index
    %a = memref.alloca() : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %cst = func.call @get() : () -> (f32)
      memref.store %cst, %a[] : memref<f32>
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
}

// CHECK:   func.func @hoist(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index) {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<f32>
// CHECK-NEXT:     memref.store %[[cst]], %[[V0]][] : memref<f32>
// CHECK-NEXT:       %[[i3:.+]] = memref.load %[[V0]][] : memref<f32>
// CHECK-NEXT:       scf.parallel (%[[arg3:.+]]) = (%[[arg1]]) to (%[[arg2]]) step (%[[c1]]) {
// CHECK-NEXT:         func.call @use(%[[i3:.+]]) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @hoist2(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index) {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<f32>
// CHECK-NEXT:     %[[i2:.+]] = arith.cmpi slt, %[[arg1]], %[[arg2]] : index
// CHECK-NEXT:     scf.if %[[i2]] {
// CHECK-NEXT:       memref.store %[[cst]], %[[V0]][] : memref<f32>
// CHECK-NEXT:       %[[i3:.+]] = memref.load %[[V0]][] : memref<f32>
// CHECK-NEXT:       scf.parallel (%[[arg3:.+]]) = (%[[arg1]]) to (%[[arg2]]) step (%[[c1]]) {
// CHECK-NEXT:         func.call @use(%[[i3:.+]]) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @hoist3(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: memref<f32>) {
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[i2:.+]] = arith.cmpi slt, %[[arg1]], %[[arg2]] : index
// CHECK-NEXT:     scf.if %[[i2]] {
// CHECK-NEXT:       %[[i3:.+]] = memref.load %[[arg3]][] : memref<f32>
// CHECK-NEXT:       scf.parallel (%[[arg4:.+]]) = (%[[arg1]]) to (%[[arg2]]) step (%[[c1]]) {
// CHECK-NEXT:         func.call @use(%[[i3:.+]]) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @nohoist(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index) {
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<f32>
// CHECK-NEXT:     scf.parallel (%[[arg3:.+]]) = (%[[arg1]]) to (%[[arg2]]) step (%[[c1]]) {
// CHECK-NEXT:       %[[V1:.+]] = func.call @get() : () -> f32
// CHECK-NEXT:       memref.store %[[V1]], %[[V0]][] : memref<f32>
// CHECK-NEXT:       %[[V2:.+]] = memref.load %[[V0]][] : memref<f32>
// CHECK-NEXT:       func.call @use(%[[V2]]) : (f32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  func.func private @use(f32)
  func.func @affhoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3 : index, %arg4 : index, %arg5: index, %arg6: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %a = memref.alloca() : memref<f32>
    memref.store %cst, %a[] : memref<f32>
    affine.parallel (%arg7, %arg8) = (max(%arg1, %arg2), %arg5) to (min(%arg3, %arg4), %arg6) {
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
  func.func @affhoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index, %arg3 : index, %arg4 : index, %arg5: index, %arg6: index, %a: memref<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    affine.parallel (%arg7, %arg8) = (max(%arg1, %arg2), %arg5) to (min(%arg3, %arg4), %arg6) {
      %v = memref.load %a[] : memref<f32>
      func.call @use(%v) : (f32) -> ()
    }
    return
  }
}

// #set = affine_set<(d0, d1, d2, d3, d4, d5) : (d3 - d0 - 1 >= 0, d3 - d1 - 1 >= 0, d4 - d0 - 1 >= 0, d4 - d1 - 1 >= 0, d5 - d2 - 1 >= 0)>

// CHECK:   func.func @affhoist(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index, %[[arg5:.+]]: index, %[[arg6:.+]]: index) {
// CHECK-NEXT:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<f32>
// CHECK-NEXT:     memref.store %[[cst]], %[[V0]][] : memref<f32>
// CHECK-NEXT:       %[[V1:.+]] = memref.load %[[V0]][] : memref<f32>
// CHECK-NEXT:       affine.parallel (%[[arg7:.+]], %[[arg8:.+]]) = (max(%[[arg1]], %[[arg2]]), %[[arg5]]) to (min(%[[arg3]], %[[arg4]]), %[[arg6]]) {
// CHECK-NEXT:         func.call @use(%[[V1]]) : (f32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @affhoist2(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: index, %[[arg2:.+]]: index, %[[arg3:.+]]: index, %[[arg4:.+]]: index, %[[arg5:.+]]: index, %[[arg6:.+]]: index, %[[arg7:.+]]: memref<f32>) {
// CHECK-NEXT:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     affine.if #set(%[[arg1]], %[[arg2]], %[[arg5]], %[[arg3]], %[[arg4]], %[[arg6]]) {
// CHECK-NEXT:       %[[V0:.+]] = memref.load %[[arg7]][] : memref<f32>
// CHECK-NEXT:       affine.parallel (%[[arg8:.+]], %[[arg9:.+]]) = (max(%[[arg1]], %[[arg2]]), %[[arg5]]) to (min(%[[arg3]], %[[arg4]]), %[[arg6]]) {
// CHECK-NEXT:         func.call @use(%[[V0]]) : (f32) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

