// RUN: cgeist %s --function=* -S | FileCheck %s

struct A {
  int value;  

  int* getPointer() {
    if (int* tmp = &this->value) { 
      return tmp; 
    }  
    return nullptr;  
  }
};

int main() {
  return *A().getPointer();
}

// CHECK:   func.func @_ZN1A10getPointerEv(
// CHECK:   "polygeist.memref2pointer"
// CHECK:   llvm.mlir.null
// CHECK:   llvm.icmp "ne"
