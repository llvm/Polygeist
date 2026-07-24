// RUN: polygeist-opt --select-func="func-name=root" %s | FileCheck %s

module {
  func.func @root(%arg0: f32) -> f32 {
    %0 = func.call @helper(%arg0) : (f32) -> f32
    return %0 : f32
  }
  func.func private @helper(%arg0: f32) -> f32 {
    %0 = func.call @logf(%arg0) : (f32) -> f32
    return %0 : f32
  }
  func.func private @logf(f32) -> f32
  func.func @unrelated() {
    return
  }
}

// CHECK: func.func @root
// CHECK: call @helper
// CHECK: func.func private @helper
// CHECK: call @logf
// CHECK: func.func private @logf
// CHECK-NOT: func.func @unrelated
