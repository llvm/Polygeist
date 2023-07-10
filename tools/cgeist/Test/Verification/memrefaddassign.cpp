// RUN: cgeist %s --function=* -c -S | FileCheck %s

float *foo(float *a) {
	a += 32;
	return a;
}
float *foo1(float *a) {
	return a + 32;
}
float *foo2(float *a) {
	return 32 + a;
}
// CHECK: func @_Z3fooPf(%[[arg0:.+]]: memref<?xf32>)
// CHECK-NEXT   %[[c32:.+]] = arith.constant 32 : index
// CHECK-NEXT   %[[V0:.+]] = "polygeist.subindex"(%[[arg0]], %[[c32]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT   return %[[V0]] : memref<?xf32>
// CHECK-NEXT }

// CHECK: func @_Z4foo1Pf(%[[arg0:.+]]: memref<?xf32>)
// CHECK-NEXT   %[[c32:.+]] = arith.constant 32 : index
// CHECK-NEXT   %[[V0:.+]] = "polygeist.subindex"(%[[arg0]], %[[c32]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT   return %[[V0]] : memref<?xf32>
// CHECK-NEXT }

// CHECK: func @_Z4foo2Pf(%[[arg0:.+]]: memref<?xf32>)
// CHECK-NEXT   %[[c32:.+]] = arith.constant 32 : index
// CHECK-NEXT   %[[V0:.+]] = "polygeist.subindex"(%[[arg0]], %[[c32]]) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT   return %[[V0]] : memref<?xf32>
// CHECK-NEXT }
