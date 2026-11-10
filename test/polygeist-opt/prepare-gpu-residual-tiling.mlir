// RUN: polygeist-opt '--prepare-gpu-residual-pipeline=function=copy tile-size=256' %s | mlir-opt --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --canonicalize | FileCheck %s
module {
  func.func @copy(%input: memref<17x19xf32>, %output: memref<17x19xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 17 : index
    %c19 = arith.constant 19 : index
    scf.parallel (%i, %j) = (%c0, %c0) to (%c7, %c19) step (%c1, %c1) {
      %value = memref.load %input[%i, %j] : memref<17x19xf32>
      memref.store %value, %output[%i, %j] : memref<17x19xf32>
    }
    return
  }
}
// CHECK-LABEL: func.func @copy
// CHECK-DAG: %[[THREADS:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[COUNT:.*]] = arith.constant 323 : index
// CHECK: gpu.launch
// CHECK-SAME: threads({{.*}} = %[[THREADS]]
// CHECK: arith.cmpi {{[su]}}lt, {{.*}}, %[[COUNT]]
// CHECK: scf.if
// CHECK: memref.load
// CHECK: memref.store
