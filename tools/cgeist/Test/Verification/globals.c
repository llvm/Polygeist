// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s

int local;
int local_init = 4;
static int internal;
static int internal_init = 5;
extern int external;

void run(int*, int*, int*, int*, int*);
void kernel_deriche() {
    run(&local, &local_init, &internal, &internal_init, &external);
}

// CHECK-DAG:   memref.global @external : memref<1xi32>
// CHECK-DAG:   memref.global "private" @internal_init : memref<1xi32> = dense<5>
// CHECK-DAG:   memref.global "private" @internal : memref<1xi32> = uninitialized
// CHECK-DAG:   memref.global @local_init : memref<1xi32> = dense<4>
// CHECK-DAG:   memref.global @local : memref<1xi32> = uninitialized
// CHECK:   func @kernel_deriche()
// CHECK-NEXT:     %[[V0:.+]] = memref.get_global @local : memref<1xi32>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %[[V2:.+]] = memref.get_global @local_init : memref<1xi32>
// CHECK-NEXT:     %[[V3:.+]] = memref.cast %[[V2]] : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %[[V4:.+]] = memref.get_global @internal : memref<1xi32>
// CHECK-NEXT:     %[[V5:.+]] = memref.cast %[[V4]] : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %[[V6:.+]] = memref.get_global @internal_init : memref<1xi32>
// CHECK-NEXT:     %[[V7:.+]] = memref.cast %[[V6]] : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     %[[V8:.+]] = memref.get_global @external : memref<1xi32>
// CHECK-NEXT:     %[[V9:.+]] = memref.cast %[[V8]] : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     call @run(%[[V1]], %[[V3]], %[[V5]], %[[V7]], %[[V9]]) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
