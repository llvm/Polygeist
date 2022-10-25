// RUN: cgeist %s --function=* -S | FileCheck %s

extern "C" {

struct pair {
    int x, y;
};
void sub0(pair& a);
void sub(pair& a) {
    a.x++;
}

void kernel_deriche() {
    pair a;
    a.x = 32;;
    pair &b = a;
    sub0(b);
}

}

// CHECK:   func @sub(%[[arg0:.+]]: memref<?x2xi32>)
// CHECK-NEXT:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:     %[[V0:.+]] = affine.load %[[arg0]][0, 0] : memref<?x2xi32>
// CHECK-NEXT:     %[[V1:.+]] = arith.addi %[[V0]], %[[c1_i32]] : i32
// CHECK-NEXT:     affine.store %[[V1]], %[[arg0]][0, 0] : memref<?x2xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func @kernel_deriche()
// CHECK-NEXT:     %[[c32_i32:.+]] = arith.constant 32 : i32
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x2xi32>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x2xi32> to memref<?x2xi32>
// CHECK-NEXT:     affine.store %[[c32_i32]], %[[V0]][0, 0] : memref<1x2xi32>
// CHECK-NEXT:     call @sub0(%[[V1]]) : (memref<?x2xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
