// RUN: polymer-opt %s -pluto-opt | FileCheck %s

func @empty() {
  return 
}

// CHECK:      func @empty() {
// CHECK-NEXT:   return
// CHECK-NEXT: }
