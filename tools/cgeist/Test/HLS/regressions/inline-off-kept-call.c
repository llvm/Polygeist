// `#pragma HLS inline off` is honoured: cgeist's inliner keeps every call to
// the function, which stays a function carrying hls.inline_off (Vitis keeps
// such a function as its own module, VITIS_NOTES N14 in the hls-est tree).
// Calls to functions without the pragma (or with `inline` on) are inlined as
// before; --hls-ignore-inline-off, and a run without --hls-annotate, inline
// every call as cgeist did before the pragma was honoured.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-ignore-inline-off -o - | FileCheck %s --check-prefix=INLINED --implicit-check-not='call @' --implicit-check-not='func.func private'
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=INLINED --implicit-check-not='call @' --implicit-check-not='func.func private'

// A callee with a local array, called twice: two calls, one function, one
// buf (inlined, each call site had its own copy of buf).
static void work(int in[1024], int out[1024]) {
#pragma HLS inline off
  int buf[1024];
  for (int i = 0; i < 1024; i++)
    buf[i] = in[i] * 3;
  for (int i = 0; i < 1024; i++)
    out[i] = buf[i] + buf[1023 - i];
}

void twice(int in0[1024], int out0[1024], int in1[1024], int out1[1024]) {
  work(in0, out0);
  work(in1, out1);
}

// A call inside a loop: the loop is still raised to affine.for and keeps the
// call (another spelling of the pragma).
static int scale(int a[16], int k) {
#pragma HLS INLINE OFF
  int s = 0;
  for (int j = 0; j < 16; j++)
    s += a[j] * k;
  return s;
}

void loop_call(int a[16], int out[8]) {
  for (int i = 0; i < 8; i++)
    out[i] = scale(a, i);
}

// `inline` (on) and no pragma: inlined; the `inline off` callee they call
// stays a call in the function they are inlined into.
static void leaf(int a[4]) {
#pragma hls inline off
  for (int i = 0; i < 4; i++)
    a[i] += 1;
}

static void mid(int a[4]) {
#pragma HLS inline
  leaf(a);
  a[0] = 7;
}

static void plain(int a[4]) { a[1] = 5; }

void nested(int a[4]) {
  mid(a);
  plain(a);
}

// An argument that does not fit the shape the parameter declares (C ignores
// the first dimension of an array parameter): the cast cgeist emits for it is
// not valid MLIR unless inlining folds it away, so this call is inlined as
// before, with a warning.
static void rows8(int p[8][4], int out[8]) {
#pragma HLS inline off
  for (int i = 0; i < 8; i++)
    out[i] = p[i][0];
}

int big[32][4];

void misfit(int out[8]) { rows8(big, out); }

// A row argument is passed as a polygeist.subindex of the array; the call is
// kept and the loop around it is raised to affine.
static void rowf(int p[4], int out[4]) {
#pragma HLS inline off
  for (int i = 0; i < 4; i++)
    out[i] = p[i] + 1;
}

void rowcall(int a[8][4], int out[8][4]) {
  for (int k = 0; k < 8; k++)
    rowf(a[k], out[k]);
}

// WARN: warning: '#pragma HLS inline off' of 'rows8' not honoured at this call: argument 0 (memref<32x4xi32>) does not fit the parameter type memref<8x4xi32>; the call is inlined
// WARN-NOT: warning

// CHECK-LABEL: func.func @twice(
// CHECK-SAME: %[[IN0:[^:]*]]: memref<1024xi32>, %[[OUT0:[^:]*]]: memref<1024xi32>, %[[IN1:[^:]*]]: memref<1024xi32>, %[[OUT1:[^:]*]]: memref<1024xi32>)
// CHECK-NEXT: call @work(%[[IN0]], %[[OUT0]])
// CHECK-NEXT: call @work(%[[IN1]], %[[OUT1]])
// CHECK-NEXT: return

// CHECK-LABEL: func.func private @work(
// CHECK-SAME: hls.inline_off
// CHECK: memref.alloca() {{.*}}polygeist.varname = "buf"
// CHECK-NOT: memref.alloca
// CHECK: affine.for
// CHECK: affine.for
// CHECK: return

// CHECK-LABEL: func.func @loop_call(
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 8 {
// CHECK-NEXT: %[[K:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK-NEXT: %[[R:.*]] = func.call @scale(%arg0, %[[K]]) : (memref<16xi32>, i32) -> i32
// CHECK-NEXT: affine.store %[[R]], %arg1[%[[I]]]

// CHECK-LABEL: func.func private @scale(
// CHECK-SAME: hls.inline_off
// CHECK: affine.for

// CHECK-LABEL: func.func @nested(
// CHECK-NOT: call @mid
// CHECK-NOT: call @plain
// CHECK: call @leaf(%arg0)
// CHECK-NOT: call
// CHECK: return
// CHECK-NOT: func.func private @mid
// CHECK-NOT: func.func private @plain

// CHECK-LABEL: func.func @misfit(
// CHECK-NOT: call
// CHECK: %[[BIG:.*]] = memref.get_global @big : memref<32x4xi32>
// CHECK: affine.for
// CHECK: affine.load %[[BIG]][%{{.*}}, 0] : memref<32x4xi32>
// CHECK-NOT: call
// CHECK: return
// CHECK-NOT: func.func private @rows8

// CHECK-LABEL: func.func @rowcall(
// CHECK-NEXT: affine.for %[[K:.*]] = 0 to 8 {
// CHECK-NEXT: %[[A:.*]] = "polygeist.subindex"(%arg0, %[[K]]) : (memref<8x4xi32>, index) -> memref<4xi32>
// CHECK-NEXT: %[[O:.*]] = "polygeist.subindex"(%arg1, %[[K]]) : (memref<8x4xi32>, index) -> memref<4xi32>
// CHECK-NEXT: func.call @rowf(%[[A]], %[[O]]) : (memref<4xi32>, memref<4xi32>) -> ()

// CHECK-LABEL: func.func private @rowf(
// CHECK-SAME: hls.inline_off
// CHECK: affine.for

// The callee of an inlined function comes last.
// CHECK-LABEL: func.func private @leaf(
// CHECK-SAME: hls.inline_off
// CHECK-NOT: func.func private @mid
// CHECK-NOT: func.func private @plain
// CHECK-NOT: func.func private @rows8

// INLINED-LABEL: func.func @twice(
// INLINED-COUNT-2: memref.alloca(){{.*}} : memref<1024xi32>
// INLINED-LABEL: func.func @loop_call(
// INLINED-LABEL: func.func @nested(
// INLINED-LABEL: func.func @misfit(
// INLINED-LABEL: func.func @rowcall(
