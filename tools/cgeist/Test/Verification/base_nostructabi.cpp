// RUN: cgeist %s --function=* --struct-abi=0 -memref-abi=0 -S | FileCheck %s

void run0(void*);
void run1(void*);
void run2(void*);

class M {
    public:
 M() { run0(this); }
};


struct _Alloc_hider : M
      {
	_Alloc_hider() { run1(this); }

      };
  
    class basic_ostringstream 
    {
    public:
      _Alloc_hider _M_stringbuf;
      basic_ostringstream() { run2(this); }
    };

void a() {
    ::basic_ostringstream a;
}

// CHECK:   func @_Z1av() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[c1_i64:.+]] = arith.constant 1 : i64
// CHECK-NEXT:     %[[V0:.+]] = llvm.alloca %[[c1_i64]] x !llvm.struct<(struct<(i8)>)> : (i64) -> !llvm.ptr<struct<(struct<(i8)>)>>
// CHECK-NEXT:     call @_ZN19basic_ostringstreamC1Ev(%[[V0]]) : (!llvm.ptr<struct<(struct<(i8)>)>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN19basic_ostringstreamC1Ev(%[[arg0:.+]]: !llvm.ptr<struct<(struct<(i8)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = llvm.getelementptr %[[arg0]][0, 0] : (!llvm.ptr<struct<(struct<(i8)>)>>) -> !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:     call @_ZN12_Alloc_hiderC1Ev(%[[V0]]) : (!llvm.ptr<struct<(i8)>>) -> ()
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[arg0]]) : (!llvm.ptr<struct<(struct<(i8)>)>>) -> memref<?xi8> 
// CHECK-NEXT:     call @_Z4run2Pv(%[[V1]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN12_Alloc_hiderC1Ev(%[[arg0:.+]]: !llvm.ptr<struct<(i8)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     call @_ZN1MC1Ev(%[[arg0]]) : (!llvm.ptr<struct<(i8)>>) -> ()
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.pointer2memref"(%[[arg0]]) : (!llvm.ptr<struct<(i8)>>) -> memref<?xi8> 
// CHECK-NEXT:     call @_Z4run1Pv(%[[V0]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func private @_Z4run2Pv(memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK-NEXT:   func @_ZN1MC1Ev(%[[arg0:.+]]: !llvm.ptr<struct<(i8)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.pointer2memref"(%[[arg0]]) : (!llvm.ptr<struct<(i8)>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z4run0Pv(%[[V0]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
