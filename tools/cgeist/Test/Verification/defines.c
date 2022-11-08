// RUN: cgeist -DOUTPUT -DTEST -c %s -S | FileCheck %s

int main(int argc, char **argv)
{
#ifdef OUTPUT
        return 1;
#else
        return 2;
#endif
}

// CHECK:   func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     return %c1_i32 : i32
// CHECK-NEXT:   }
