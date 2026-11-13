// RUN: %python %S/../../scripts/correctness/add_repetition_session.py %s --function=apply --session=apply_many -o %t
// RUN: FileCheck %s < %t

module {
  func.func @apply(%arg0: memref<?xf64>, %arg1: memref<4x?xf64>) {
    return
  }
}

// CHECK: func.func @apply_many(%repetitions: i32, %arg0: memref<?xf64>, %arg1: memref<4x?xf64>)
// CHECK: %[[COUNT:.*]] = arith.index_cast %repetitions : i32 to index
// CHECK: scf.for {{.*}} = {{.*}} to %[[COUNT]] step {{.*}} {
// CHECK: func.call @apply(%arg0, %arg1) : (memref<?xf64>, memref<4x?xf64>) -> ()
