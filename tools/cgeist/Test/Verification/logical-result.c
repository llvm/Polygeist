// RUN: cgeist %s --function=logical_and -S | FileCheck %s --check-prefix=AND
// RUN: cgeist %s --function=logical_or -S | FileCheck %s --check-prefix=OR
// RUN: cgeist %s --function=logical_and -S -O3 | FileCheck %s --check-prefix=AND
// RUN: cgeist %s --function=logical_or -S -O3 | FileCheck %s --check-prefix=OR
int logical_and(int a, int b) { return a && b; }
int logical_or(int a, int b) { return a || b; }
// AND-LABEL: func.func @logical_and
// AND-NOT: arith.extsi
// AND: arith.extui {{.*}} : i1 to i32
// AND-NOT: arith.extsi
// OR-LABEL: func.func @logical_or
// OR-NOT: arith.extsi
// OR: arith.extui {{.*}} : i1 to i32
// OR-NOT: arith.extsi
