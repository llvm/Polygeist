// RUN: cgeist %s --function=* -S | FileCheck %s

class M {
};

struct _Alloc_hider : M
      {
	_Alloc_hider() { }

	char* _M_p; // The actual data.
      };


    class basic_streambuf
    {
  public:
      /// Destructor deallocates no buffer space.
      virtual
      ~basic_streambuf()
      { }
    };
    class mbasic_stringbuf : public basic_streambuf

    {
    public:

      _Alloc_hider	_M_dataplus;
      mbasic_stringbuf()
      { }

    };

void a() {
    mbasic_stringbuf a;
}
// CHECK-LABEL:   func.func @_Z1av()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN16mbasic_stringbufC1Ev(
// CHECK-SAME:                                         %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(ptr)>, !llvm.struct<(struct<(i8)>, memref<?xi8>)>)>>)
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN15basic_streambufC1Ev(
// CHECK-SAME:                                        %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(ptr)>>)
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN12_Alloc_hiderC1Ev(
// CHECK-SAME:                                     %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(i8)>, memref<?xi8>)>>)
// CHECK:           return
// CHECK:         }

