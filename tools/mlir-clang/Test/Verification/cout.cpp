// RUN: mlir-clang %s --function=* -S | FileCheck %s
#include <iostream>

void moo(int x) {
    std::cout << x << std::endl;
}
