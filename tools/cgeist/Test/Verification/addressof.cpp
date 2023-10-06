// RUN: cgeist %s %stdinclude --function=* -S | FileCheck %s

#include <memory>
 
struct Ptr {
};
 
Ptr *foo()
{
	Ptr p;
	return std::addressof(p);  // calls Ptr<int>* overload, (= this)
}

Ptr *bar()
{
	Ptr p;
	return __builtin_addressof(p);  // calls Ptr<int>* overload, (= this)
}

// CHECK-LABEL:   func.func @_Z3foov() -> memref<?x!llvm.struct<(i8)>>  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(i8)>>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.cast %[[VAL_0]] : memref<1x!llvm.struct<(i8)>> to memref<?x!llvm.struct<(i8)>>
// CHECK:           return %[[VAL_1]] : memref<?x!llvm.struct<(i8)>>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3barv() -> memref<?x!llvm.struct<(i8)>>  
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(i8)>>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.cast %[[VAL_0]] : memref<1x!llvm.struct<(i8)>> to memref<?x!llvm.struct<(i8)>>
// CHECK:           return %[[VAL_1]] : memref<?x!llvm.struct<(i8)>>
// CHECK:         }

