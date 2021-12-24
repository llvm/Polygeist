// RUN: mlir-clang %s --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

struct MOperandInfo* begin();

struct MOperandInfo& inner() {
  return begin()[0];
}

