// RUN: cgeist %s --function=* -S | FileCheck %s

struct Node {
  struct Node *next;
  int value;
};

struct Node root;

int read_root(void) { return root.value; }

// CHECK: memref.global @root : memref<1x!llvm.struct<"opaque@polygeist@mlir@struct.Node"
// CHECK-LABEL: func.func @read_root
// CHECK: memref.get_global @root : memref<1x!llvm.struct<"opaque@polygeist@mlir@struct.Node"
