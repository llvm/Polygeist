// RUN: cgeist %s --function=* -S | FileCheck %s

int get();
void other();

int checkCmdLineFlag(const int argc) {
  int bFound = 0;

    for (int i = 1; i < argc; i++) {
      if (get()) {
        bFound = 1;
        continue;
      }
      other();
    }

  return bFound;
}

// CHECK-LABEL:   func.func @checkCmdLineFlag(
// CHECK-SAME:                                %[[VAL_0:[A-Za-z0-9_]*]]: i32) -> i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = scf.for %[[VAL_6:[A-Za-z0-9_]*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_1]] iter_args(%[[VAL_7:[A-Za-z0-9_]*]] = %[[VAL_3]]) -> (i32) {
// CHECK:             %[[VAL_8:[A-Za-z0-9_]*]] = func.call @get() : () -> i32
// CHECK:             %[[VAL_9:[A-Za-z0-9_]*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_10:[A-Za-z0-9_]*]] = arith.select %[[VAL_9]], %[[VAL_2]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_11:[A-Za-z0-9_]*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:             scf.if %[[VAL_11]] {
// CHECK:               func.call @other() : () -> ()
// CHECK:             }
// CHECK:             scf.yield %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }
// CHECK:         func.func private @get() -> i32
// CHECK:         func.func private @other()
