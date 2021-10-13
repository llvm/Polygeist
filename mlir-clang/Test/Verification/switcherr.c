// RUN: mlir-clang %s --function=foo -S | FileCheck %s

int foo(int t) {
  switch (t) {
  case 1:
    //n = 20;
    break;
    /*
  case 2:
    n = 30;
    break;
    */
  default:
    return -1;
  }
  return 10;
}
