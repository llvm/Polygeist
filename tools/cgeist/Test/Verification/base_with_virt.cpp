// RUN: cgeist %s --function=* -S | FileCheck %s

class M {
};


struct _Alloc_hider : M
      {
  _Alloc_hider() {}

  char *_M_p; // The actual data.
      };

class A {
    public:
  int x;
  virtual void foo();
  A() : x(3) {}
};

class mbasic_stringbuf : public A {
public:
  _Alloc_hider _M_dataplus;
  mbasic_stringbuf() {}
};

void a() {
    mbasic_stringbuf a;
}

// CHECK:   func.func @_Z1av() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<1x!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>> to memref<?x!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>
// CHECK-NEXT: call @_ZN16mbasic_stringbufC1Ev(%[[V1]]) : (memref<?x!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK: func.func @_ZN16mbasic_stringbufC1Ev(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>) -> !llvm.ptr<!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<!llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>) -> memref<?x!llvm.struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>
// CHECK-NEXT:     call @_ZN1AC1Ev(%[[V1]]) : (memref<?x!llvm.struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN1AC1Ev(%[[arg0:.+]]: memref<?x!llvm.struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) -> !llvm.ptr<struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 1] : (!llvm.ptr<struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %[[c3_i32]], %[[V1]] : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN12_Alloc_hiderC1Ev(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(i8)>, memref<?xi8>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     return
// CHECK-NEXT:   }
