// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

void use(int i);

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int start, int end) {
  for (int i = end; i >= start; i--) {
    use(i);
  }
}

// CHECK: #map = affine_map<()[s0] -> (s0 + 1)>
// CHECK: kernel_correlation
// CHECK-DAG:     %[[Cm1:.+]] = arith.constant -1 : index
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT:     %[[V1:.+]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT:     affine.for %[[arg2:.+]] = %[[V1]] to #map()[%[[V0]]] {
// CHECK-NEXT:       %[[V2:.+]] = arith.subi %[[arg2]], %[[V1]] : index
// CHECK-NEXT:       %[[V3:.+]] = arith.muli %[[V2]], %[[Cm1]] : index
// CHECK-NEXT:       %[[V4:.+]] = arith.addi %[[V0]], %[[V3]] : index
// CHECK-NEXT:       %[[V5:.+]] = arith.index_cast %[[V4]] : index to i32
// CHECK-NEXT:       call @use(%[[V5]]) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
