// RUN: cgeist %s --function=kernel_deriche -S | FileCheck %s

int deref(int a);

void kernel_deriche(int *a) {
    deref(*a);
}

// CHECK:    func @kernel_deriche(%[[arg0:.+]]: memref<?xi32>)
// CHECK-NEXT:    %[[V0:.+]] = affine.load %[[arg0]][0] : memref<?xi32>
// CHECK-NEXT:    %[[V1:.+]] = call @deref(%[[V0]]) : (i32) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }
