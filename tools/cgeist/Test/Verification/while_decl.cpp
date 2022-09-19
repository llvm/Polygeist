// RUN: cgeist %s --function=* -S | FileCheck %s

struct A {
  int value;  

  int getValue() {
    while (int tmp = this->value) { 
      tmp++;
    }  
    return value;  
  }
};

int main() {
  return A().getValue();
}

// CHECK:   func.func @_ZN1A8getValueEv(
// CHECK:   scf.while
// CHECK:   arith.cmpi ne
// CHECK:   scf.condition
