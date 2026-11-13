// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=triangular' %s | FileCheck %s
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=triangular' %s | polygeist-opt '--prepare-gpu-residual-pipeline=function=triangular' | FileCheck %s
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=cross_row' %s | FileCheck %s --check-prefix=DEPENDENT
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=multiple_writes assume-no-alias=true' %s | FileCheck %s --check-prefix=DISJOINT
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=multiple_writes' %s | FileCheck %s --check-prefix=MAYALIAS
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=scalar_write' %s | FileCheck %s --check-prefix=SCALAR
// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=host_stack' %s | FileCheck %s --check-prefix=HOST-STACK
module {
  func.func @triangular(%matrix: memref<?x?xf64>, %n: index, %scale: f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %row = %c1 to %n step %c1 {
      scf.for %col = %c0 to %row step %c1 {
        %value = memref.load %matrix[%row, %col] : memref<?x?xf64>
        %scaled = arith.divf %value, %scale : f64
        memref.store %scaled, %matrix[%row, %col] : memref<?x?xf64>
      }
    }
    scf.for %row = %c0 to %n step %c1 {
      %start = arith.addi %row, %c1 : index
      scf.for %col = %start to %n step %c1 {
        %value = memref.load %matrix[%row, %col] : memref<?x?xf64>
        %scaled = arith.divf %value, %scale : f64
        memref.store %scaled, %matrix[%row, %col] : memref<?x?xf64>
      }
    }
    return
  }
  func.func @cross_row(%matrix: memref<?x?xf64>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %row = %c1 to %n step %c1 {
      scf.for %col = %c0 to %row step %c1 {
        %value = memref.load %matrix[%col, %row] : memref<?x?xf64>
        memref.store %value, %matrix[%row, %col] : memref<?x?xf64>
      }
    }
    return
  }
  func.func @multiple_writes(%matrix: memref<?x?xf64>, %bias: memref<?xf64>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %row = %c0 to %n step %c1 {
      scf.for %col = %c0 to %n step %c1 {
        %value = memref.load %matrix[%row, %col] : memref<?x?xf64>
        %offset = memref.load %bias[%col] : memref<?xf64>
        %added = arith.addf %value, %offset : f64
        memref.store %added, %matrix[%row, %col] : memref<?x?xf64>
        %scaled = arith.mulf %added, %offset : f64
        memref.store %scaled, %matrix[%row, %col] : memref<?x?xf64>
      }
    }
    return
  }
  func.func @scalar_write(%matrix: memref<?x?xf64>, %index: index, %value: f64) {
    memref.store %value, %matrix[%index, %index] : memref<?x?xf64>
    return
  }
  func.func @host_stack(%value: f32) -> f32 {
    %scratch = memref.alloca() : memref<f32>
    memref.store %value, %scratch[] : memref<f32>
    %result = memref.load %scratch[] : memref<f32>
    return %result : f32
  }
}
// CHECK-LABEL: func.func @triangular
// CHECK: scf.parallel (%[[ROW:[^)]+]]) =
// CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %[[ROW]]
// CHECK: arith.divf
// CHECK: memref.store
// CHECK: scf.parallel (%[[ROW2:[^)]+]]) =
// CHECK-NEXT: %[[START:.*]] = arith.addi %[[ROW2]], %{{.*}}
// CHECK-NEXT: scf.for %{{.*}} = %[[START]]
// CHECK: return
// DEPENDENT-LABEL: func.func @cross_row
// DEPENDENT-NOT: scf.parallel
// DEPENDENT: scf.for
// DEPENDENT-NEXT: scf.for
// DEPENDENT: memref.load
// DEPENDENT: memref.store
// DEPENDENT: return
// DISJOINT-LABEL: func.func @multiple_writes
// DISJOINT: scf.parallel
// DISJOINT-NEXT: scf.for
// DISJOINT: memref.store
// DISJOINT: memref.store
// DISJOINT: return
// MAYALIAS-LABEL: func.func @multiple_writes
// MAYALIAS-NOT: scf.parallel
// MAYALIAS: scf.for
// MAYALIAS-NEXT: scf.for
// MAYALIAS: return
// SCALAR-LABEL: func.func @scalar_write
// SCALAR: scf.parallel
// SCALAR-NEXT: memref.store
// SCALAR: return
// HOST-STACK-LABEL: func.func @host_stack
// HOST-STACK-NOT: scf.parallel
// HOST-STACK: memref.store
// HOST-STACK-NEXT: memref.load
// HOST-STACK: return
