// RUN: cgeist %s --function=* -S | FileCheck %s

static int flat[80];

void consume(int rows[][8]);

void flat_to_ranked_pointer(void) {
  consume((int (*)[8])(void *)flat);
}

// CHECK-LABEL: func.func @flat_to_ranked_pointer
// CHECK: %[[FLAT:.+]] = memref.get_global @flat : memref<80xi32>
// CHECK: %[[PTR0:.+]] = "polygeist.memref2pointer"(%[[FLAT]]) : (memref<80xi32>) -> !llvm.ptr
// CHECK: %[[RANKED:.+]] = "polygeist.pointer2memref"(%[[PTR0]]) : (!llvm.ptr) -> memref<?x8xi32>
// CHECK: call @consume(%[[RANKED]]) : (memref<?x8xi32>) -> ()
