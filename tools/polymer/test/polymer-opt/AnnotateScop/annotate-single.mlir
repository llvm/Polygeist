// RUN: polymer-opt %s -annotate-scop="functions=foo" | FileCheck %s

func @foo() {
  return
} 
 
func @bar() {
  return
} 

// CHECK: func @foo() {
// CHECK: func @bar() attributes {scop.ignored} {
