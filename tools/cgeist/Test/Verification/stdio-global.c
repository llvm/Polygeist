// RUN: cgeist %s %stdinclude --function=emit -S | FileCheck %s

#include <stdio.h>

void emit(void) { fprintf(stderr, "ok\n"); }

// CHECK: llvm.mlir.global external @stderr() {{.*}} : !llvm.ptr
// CHECK: %[[ADDRESS:.+]] = llvm.mlir.addressof @stderr : !llvm.ptr
// CHECK: %[[STREAM:.+]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @fprintf(%[[STREAM]],
