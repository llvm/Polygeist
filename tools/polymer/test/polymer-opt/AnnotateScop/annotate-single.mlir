// RUN: polymer-opt %s -annotate-scop="functions=foo" | FileCheck %s

func.func @foo() {
  return
} 
 
func.func @bar() {
  return
} 

// CHECK: func.func @foo() {
// CHECK: func.func @bar() attributes {scop.ignored} {
