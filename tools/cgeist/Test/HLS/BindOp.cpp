//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

int foo (int a, int b) {
    int c, d;
#pragma HLS BIND_OP variable=c op=mul impl=fabric latency=2
    c = a*b;
    d = a*c;
    return d;
}

// The pragma binds the multiplication assigned to c (not the one assigned to d).
// CHECK: %[[C:.*]] = arith.muli %arg0, %arg1 {hls.bind_op = [{impl = "fabric", latency = 2 : i32, op = "mul", variable = "c"}]} : i32
// CHECK-NEXT: arith.muli %arg0, %[[C]] : i32