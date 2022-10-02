// RUN: cgeist %s --function=* -S | FileCheck %s

void free(void*);

void metafree(void* x, void (*foo)(int), void (*bar)()) {
    foo(0);
    bar();
    free(x);
}

// CHECK:   func.func @metafree(%arg0: memref<?xi8>, %arg1: memref<?x!llvm.func<void (i32)>>, %arg2: memref<?x!llvm.func<void (...)>>)
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg1) : (memref<?x!llvm.func<void (i32)>>) -> !llvm.ptr<func<void (i32)>>
// CHECK-NEXT:     llvm.call %0(%c0_i32) : (i32) -> ()
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%arg2) : (memref<?x!llvm.func<void (...)>>) -> !llvm.ptr<func<void (...)>>
// CHECK-NEXT:     llvm.call %1() : () -> ()
// CHECK-NEXT:     memref.dealloc %arg0 : memref<?xi8>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
