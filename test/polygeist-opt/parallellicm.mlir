// RUN: polygeist-opt --parallel-licm --split-input-file %s | FileCheck %s

module {
  func private @use(f32) 
  func @hoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    memref.store %cst, %a[] : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %v = memref.load %a[] : memref<f32>
      call @use(%v) : (f32) -> ()
    }
    return
  }
  func @hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      memref.store %cst, %a[] : memref<f32>
      %v = memref.load %a[] : memref<f32>
      call @use(%v) : (f32) -> ()
    }
    return
  }
  func private @get() -> (f32) 
  func @nohoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
    %c1 = arith.constant 1 : index 
    %a = memref.alloca() : memref<f32>
    scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
      %cst = call @get() : () -> (f32)
      memref.store %cst, %a[] : memref<f32>
      %v = memref.load %a[] : memref<f32>
      call @use(%v) : (f32) -> ()
    }
    return
  }
}

// CHECK:   func @hoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = memref.alloca() : memref<f32>
// CHECK-NEXT:     memref.store %cst, %0[] : memref<f32>
// CHECK-NEXT:     %1 = arith.cmpi slt, %arg1, %arg2 : index
// CHECK-NEXT:     scf.if %1 {
// CHECK-NEXT:       %2 = memref.load %0[] : memref<f32>
// CHECK-NEXT:       scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
// CHECK-NEXT:         call @use(%2) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func @hoist2(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = memref.alloca() : memref<f32>
// CHECK-NEXT:     %1 = arith.cmpi slt, %arg1, %arg2 : index
// CHECK-NEXT:     scf.if %1 {
// CHECK-NEXT:       memref.store %cst, %0[] : memref<f32>
// CHECK-NEXT:       %2 = memref.load %0[] : memref<f32>
// CHECK-NEXT:       scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
// CHECK-NEXT:         call @use(%2) : (f32) -> ()
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func @nohoist(%arg0: memref<?xf32>, %arg1: index, %arg2: index) {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = memref.alloca() : memref<f32>
// CHECK-NEXT:     scf.parallel (%arg3) = (%arg1) to (%arg2) step (%c1) {
// CHECK-NEXT:       %1 = call @get() : () -> f32
// CHECK-NEXT:       memref.store %1, %0[] : memref<f32>
// CHECK-NEXT:       %2 = memref.load %0[] : memref<f32>
// CHECK-NEXT:       call @use(%2) : (f32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
