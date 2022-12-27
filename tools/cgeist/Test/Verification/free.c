// RUN: cgeist %s --function=* -S | FileCheck %s

void free(void*);

int* metafree(void* x, void (*foo)(int), void (*bar)(), int* g(int*), int* h) {
    foo(0);
    bar();
    free(x);
    return g(h);
}

// CHECK:   func.func @metafree(%[[arg0:.+]]: memref<?xi8>, %[[arg1:.+]]: memref<?x!llvm.func<void (i32)>>, %[[arg2:.+]]: memref<?x!llvm.func<void (...)>>, %arg3: memref<?x!llvm.func<memref<?xi32> (memref<?xi32>)>>, %arg4: memref<?xi32>) -> memref<?xi32>
// CHECK-NEXT:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg1]]) : (memref<?x!llvm.func<void (i32)>>) -> !llvm.ptr<func<void (i32)>>
// CHECK-NEXT:     llvm.call %[[V0]](%[[c0_i32]]) : (i32) -> ()
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[arg2]]) : (memref<?x!llvm.func<void (...)>>) -> !llvm.ptr<func<void (...)>>
// CHECK-NEXT:     llvm.call %[[V1]]() : () -> ()
// CHECK-NEXT:     memref.dealloc %[[arg0]] : memref<?xi8>
// CHECK-NEXT:     %[[fn:.+]] = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.func<memref<?xi32> (memref<?xi32>)>>) -> !llvm.ptr<func<ptr<i32> (ptr<i32>)>>
// CHECK-NEXT:     %[[inp:.+]] = "polygeist.memref2pointer"(%arg4) : (memref<?xi32>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[cal:.+]] = llvm.call %[[fn]](%[[inp]]) : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[res:.+]] = "polygeist.pointer2memref"(%[[cal]]) : (!llvm.ptr<i32>) -> memref<?xi32>
// CHECK-NEXT:     return %[[res]] : memref<?xi32>
// CHECK-NEXT:   }
