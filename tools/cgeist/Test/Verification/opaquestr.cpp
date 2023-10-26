// RUN: cgeist %s %stdinclude -S | FileCheck %s

struct Str {
  int a;
  float b;
};

struct OperandInfo {
  OperandInfo *info;
  struct {
    int a;
    float b;
  } intfloat;
  Str a;
  Str *b;
  int c;
};

void *foo(OperandInfo *info) {
  return info;
}

// CHECK: memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.OperandInfo", (memref<?x!llvm.struct<"opaque@polygeist@mlir@struct.OperandInfo">>, struct<(i32, f32)>, struct<(i32, f32)>, memref<?x!llvm.struct<(i32, f32)>>, i32)>>
