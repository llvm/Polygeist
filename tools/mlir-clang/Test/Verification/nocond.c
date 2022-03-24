// RUN: mlir-clang %s --function=* -S | FileCheck %s

int run();

void what() {
  for (;;) {
    if (run()) break;
  }
}


