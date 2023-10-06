// RUN: polygeist-opt --canonicalize-polygeist --split-input-file %s | FileCheck %s

// -----

// CHECK:  func.func @main(%[[arg0:.+]]: index) -> memref<1000xi32> {
// CHECK:    %[[V0:.+]] = memref.alloca() : memref<2x1000xi32>
// CHECK:    %[[V1:.+]] = "polygeist.subindex"(%[[V0]], %[[arg0]]) : (memref<2x1000xi32>, index) -> memref<1000xi32>
// CHECK:    return %[[V1]] : memref<1000xi32>
// CHECK:  }
func.func @main(%arg0 : index) -> memref<1000xi32> {
  %c0 = arith.constant 0 : index
  %1 = memref.alloca() : memref<2x1000xi32>
    %3 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) -> memref<?x1000xi32>
    %4 = "polygeist.subindex"(%3, %c0) : (memref<?x1000xi32>, index) -> memref<1000xi32>
  return %4 : memref<1000xi32>
}

// -----

  func.func @fold2ref(%arg0 : !llvm.ptr) -> memref<?xi32> {
        %c0_i32 = arith.constant 0 : i32
        %11 = llvm.getelementptr %arg0[%c0_i32, 0] {elem_type = !llvm.struct<(i32, i32)>} : (!llvm.ptr, i32) -> !llvm.ptr
        %12 = "polygeist.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi32>
    return %12 : memref<?xi32>
  }

// CHECK:   func.func @fold2ref(%[[arg0:.+]]: !llvm.ptr) -> memref<?xi32> {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.pointer2memref"(%[[arg0]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:     return %[[V0]] : memref<?xi32>
// CHECK-NEXT:   }

  func.func @nofold2ref(%arg0 : !llvm.ptr) -> memref<?xi32> {
        %c0_i32 = arith.constant 0 : i32
        %11 = llvm.getelementptr %arg0[%c0_i32, 1] {elem_type = !llvm.struct<(i32, i32)>} : (!llvm.ptr, i32) -> !llvm.ptr
        %12 = "polygeist.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi32>
    return %12 : memref<?xi32>
  }

// CHECK: @nofold2ref(%[[arg0:.+]]: !llvm.ptr) -> memref<?xi32> {
// CHECK-NEXT:     %[[V0:.+]] = llvm.getelementptr %[[arg0]][0, 1] : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:     return %[[V1]] : memref<?xi32>
// CHECK-NEXT:   }

func.func @memref2ptr(%arg0: memref<10xi32>) -> !llvm.ptr {
     %c2 = arith.constant 2 : index
     %0 = "polygeist.subindex"(%arg0, %c2) : (memref<10xi32>, index) -> memref<?xi32>
     %1 = "polygeist.memref2pointer"(%0) : (memref<?xi32>) -> !llvm.ptr
     return %1 : !llvm.ptr
}
// CHECK: func.func @memref2ptr(%[[arg0:.+]]: memref<10xi32>) -> !llvm.ptr {
// CHECK-NEXT: %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<10xi32>) -> !llvm.ptr
// CHECK-NEXT: %[[V1:.+]] = llvm.getelementptr %[[V0]][8] : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: return %[[V1]] : !llvm.ptr
// CHECK-NEXT: }

module {
func.func private @wow0()
func.func private @wow1()
func.func private @wow2()
func.func private @wow3()
func.func private @wow4()
func.func @flatten_alternatives() {
  "polygeist.alternatives"() ({
    func.call @wow0() : () -> ()
    "polygeist.alternatives"() ({
      func.call @wow1() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      func.call @wow2() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["1","2"]} : () -> ()
    "polygeist.polygeist_yield"() : () -> ()
  }, {
    "polygeist.alternatives"() ({
      func.call @wow3() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }, {
      func.call @wow4() : () -> ()
      "polygeist.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["3","4"]} : () -> ()
    "polygeist.polygeist_yield"() : () -> ()
  }) {alternatives.descs = ["a","b"]} : () -> ()
  return
}
}
// CHECK:  func.func @flatten_alternatives() {
// CHECK-NEXT:    "polygeist.alternatives"() ({
// CHECK-NEXT:      func.call @wow3() : () -> ()
// CHECK-NEXT:      "polygeist.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      func.call @wow4() : () -> ()
// CHECK-NEXT:      "polygeist.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      func.call @wow0() : () -> ()
// CHECK-NEXT:      func.call @wow1() : () -> ()
// CHECK-NEXT:      "polygeist.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      func.call @wow0() : () -> ()
// CHECK-NEXT:      func.call @wow2() : () -> ()
// CHECK-NEXT:      "polygeist.polygeist_yield"() : () -> ()
// CHECK-NEXT:      }) {alternatives.descs = ["b3", "b4", "a1", "a2"]} : () -> ()
