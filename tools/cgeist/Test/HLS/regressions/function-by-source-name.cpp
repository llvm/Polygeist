// `--function=<name>` selects a C++ function by its source name (the HLS top
// as Vitis' set_top names it) as well as by its mangled name. cgeist used to
// match only the mangled name, so a C++ top given by name produced an empty
// module (class_ece8893_2023_spring FocusedACO, whose `--function='*'` run
// stops in a function outside the top's call graph, W8b).
// RUN: cgeist %s %resourcedir --function=top -S -o - | FileCheck %s
// RUN: cgeist %s %resourcedir --function=ns::inner -S -o - | FileCheck %s --check-prefix=QUAL
// RUN: cgeist %s %resourcedir --function=_Z3topii -S -o - | FileCheck %s

// CHECK: func.func @_Z3topii(
// CHECK-NOT: func.func @_Z5otheri(
// QUAL: func.func @_ZN2ns5innerEi(
// QUAL-NOT: func.func @_Z3topii(
namespace ns {
int inner(int a) { return a * 3; }
} // namespace ns

int top(int first, int length) { return first > 0 ? first - 1 : length - 1; }

int other(int a) { return a + 1; }
