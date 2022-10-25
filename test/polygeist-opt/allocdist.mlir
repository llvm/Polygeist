// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func.func private @capture(%a : memref<i32>) 
  func.func private @use(%a : memref<?xi32>, %b : f32, %d : i32, %e : f32)
  func.func @main() {
    %c0 = arith.constant 0 : index
    %cc1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg2) = (%c0) to (%c5) step (%cc1) {
      %a1 = memref.alloca() : memref<2xi32>
      %a2 = memref.cast %a1 : memref<2xi32> to memref<?xi32>
      %b1 = memref.alloca() : memref<f32>
      %c1 = memref.alloca() : memref<i32>
      %d1 = memref.alloca() : memref<1xi32>
      %b2 = memref.load %b1[] : memref<f32>
      func.call @capture(%c1) : (memref<i32>) -> ()
      %d2 = memref.cast %d1 : memref<1xi32> to memref<?xi32>

      %e1 = memref.alloca() : memref<1xf32>
      %e2 = memref.cast %e1 : memref<1xf32> to memref<?xf32>
      %e3 = memref.load %e2[%c0] : memref<?xf32>

      "polygeist.barrier"(%arg2) : (index) -> ()

      %d3 = memref.load %d2[%c0] : memref<?xi32>
      func.call @use(%a2, %b2, %d3, %e3) : (memref<?xi32>, f32, i32, f32) -> ()
      scf.yield
    }
    return
  }
}

// CHECK:   func.func @main() {
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[c5:.+]] = arith.constant 5 : index
// CHECK-NEXT:     memref.alloca_scope {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca(%[[c5]]) : memref<?xf32>
// CHECK-NEXT:     %[[V1:.+]] = memref.alloca(%[[c5]]) : memref<?xmemref<?xi32>>
// CHECK-NEXT:     %[[V2:.+]] = memref.alloca(%[[c5]]) : memref<?xf32>
// CHECK-NEXT:     %[[V3:.+]] = memref.alloca(%[[c5]]) : memref<?xmemref<?xi32>>
// CHECK-NEXT:     %[[V4:.+]] = memref.alloca(%[[c5]]) : memref<?x2xi32>
// CHECK-NEXT:     %[[V5:.+]] = memref.alloca(%[[c5]]) : memref<?xi32>
// CHECK-NEXT:     %[[V6:.+]] = memref.alloca(%[[c5]]) : memref<?x1xi32>
// CHECK-NEXT:     scf.parallel (%[[arg0:.+]]) = (%[[c0]]) to (%[[c5]]) step (%[[c1]]) {
// CHECK-NEXT:       %[[V7:.+]] = "polygeist.subindex"(%[[V4]], %[[arg0]]) : (memref<?x2xi32>, index) -> memref<2xi32>
// CHECK-NEXT:       %[[V8:.+]] = memref.cast %[[V7]] : memref<2xi32> to memref<?xi32>
// CHECK-NEXT:       memref.store %[[V8]], %[[V3]][%[[arg0]]] : memref<?xmemref<?xi32>>
// CHECK-NEXT:       %[[V9:.+]] = memref.alloca() : memref<f32>
// CHECK-NEXT:       %[[V10:.+]] = memref.load %[[V9]][] : memref<f32>
// CHECK-NEXT:       memref.store %[[V10]], %[[V2]][%[[arg0]]] : memref<?xf32>
// CHECK-NEXT:       %[[V11:.+]] = "polygeist.subindex"(%[[V5]], %[[arg0]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:       func.call @capture(%[[V11]]) : (memref<i32>) -> ()
// CHECK-NEXT:       %[[V12:.+]] = "polygeist.subindex"(%[[V6]], %[[arg0]]) : (memref<?x1xi32>, index) -> memref<1xi32>
// CHECK-NEXT:       %[[V13:.+]] = memref.cast %[[V12]] : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:       memref.store %[[V13]], %[[V1]][%[[arg0]]] : memref<?xmemref<?xi32>>
// CHECK-NEXT:       %[[V14:.+]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:       %[[V15:.+]] = memref.load %[[V14]][%[[c0]]] : memref<1xf32>
// CHECK-NEXT:       memref.store %[[V15]], %[[V0]][%[[arg0]]] : memref<?xf32>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.parallel (%[[arg0:.+]]) = (%[[c0]]) to (%[[c5]]) step (%[[c1]]) {
// CHECK-DAG:       %[[i7:.+]] = memref.load %[[V1]][%[[arg0]]] : memref<?xmemref<?xi32>>
// CHECK-DAG:       %[[i8:.+]] = memref.load %[[i7]][%[[c0]]] : memref<?xi32>
// CHECK-DAG:       %[[i9:.+]] = memref.load %[[V3]][%[[arg0]]] : memref<?xmemref<?xi32>>
// CHECK-DAG:       %[[i10:.+]] = memref.load %[[V2]][%[[arg0]]] : memref<?xf32>
// CHECK-DAG:       %[[i11:.+]] = memref.load %[[V0]][%[[arg0]]] : memref<?xf32>
// CHECK-DAG:       func.call @use(%[[i9]], %[[i10]], %[[i8]], %[[i11]]) : (memref<?xi32>, f32, i32, f32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
