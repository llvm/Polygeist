// RUN: cgeist %s --function=* -S | FileCheck %s

void free(void*);

int* metafree(void* x, void (*foo)(int), void (*bar)(), int* g(int*), int* h) {
    foo(0);
    bar();
    free(x);
    return g(h);
}

// CHECK-LABEL:   func.func @metafree(
// CHECK-SAME:                        %[[VAL_0:[A-Za-z0-9_]*]]: memref<?xi8>,
// CHECK-SAME:                        %[[VAL_1:[A-Za-z0-9_]*]]: memref<?x!llvm.func<void (i32)>>,
// CHECK-SAME:                        %[[VAL_2:[A-Za-z0-9_]*]]: memref<?x!llvm.func<void (...)>>,
// CHECK-SAME:                        %[[VAL_3:[A-Za-z0-9_]*]]: memref<?x!llvm.func<memref<?xi32> (memref<?xi32>)>>,
// CHECK-SAME:                        %[[VAL_4:[A-Za-z0-9_]*]]: memref<?xi32>) -> memref<?xi32>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?x!llvm.func<void (i32)>>) -> !llvm.ptr
// CHECK:           llvm.call %[[VAL_6]](%[[VAL_5]]) : !llvm.ptr, (i32) -> ()
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<?x!llvm.func<void (...)>>) -> !llvm.ptr
// CHECK:           llvm.call %[[VAL_7]]() : !llvm.ptr, () -> ()
// CHECK:           memref.dealloc %[[VAL_0]] : memref<?xi8>
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_3]]) : (memref<?x!llvm.func<memref<?xi32> (memref<?xi32>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<?xi32>) -> !llvm.ptr
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = llvm.call %[[VAL_8]](%[[VAL_9]]) : !llvm.ptr, (!llvm.ptr) -> !llvm.ptr
// CHECK:           %[[VAL_11:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_10]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK:           return %[[VAL_11]] : memref<?xi32>
// CHECK:         }

