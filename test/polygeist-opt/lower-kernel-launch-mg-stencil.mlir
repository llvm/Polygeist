// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @customMGResid_f64_memref(
      %u: memref<?xi8>, %v: memref<?xi8>, %r: memref<?xi8>,
      %n1: i32, %n2: i32, %n3: i32, %a: memref<?xf64>) {
    kernel.yield
  }
  kernel.defn @customMGPSInv_f64_memref(
      %r: memref<?xi8>, %u: memref<?xi8>,
      %n1: i32, %n2: i32, %n3: i32, %c: memref<?xf64>) {
    kernel.yield
  }

  func.func @mg(%u: memref<?xi8>, %v: memref<?xi8>, %r: memref<?xi8>,
                %n1: i32, %n2: i32, %n3: i32,
                %a: memref<?xf64>, %c: memref<?xf64>) {
    kernel.launch @customMGResid_f64_memref(
        %u, %v, %r, %n1, %n2, %n3, %a)
        : (memref<?xi8>, memref<?xi8>, memref<?xi8>, i32, i32, i32,
           memref<?xf64>) -> ()
    kernel.launch @customMGPSInv_f64_memref(
        %r, %u, %n1, %n2, %n3, %c)
        : (memref<?xi8>, memref<?xi8>, i32, i32, i32, memref<?xf64>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @mg
// CHECK: call @polygeist_mg_resid_f64
// CHECK: call @polygeist_mg_psinv_f64
// CHECK-NOT: kernel.launch
