// A struct whose members share one type is a memref of that type with the
// members as its last dimension (CombinedStructABI): `node_t nodes[16]` with
// `struct { uint64_t edge_begin, edge_end; }` is memref<16x2xi64>, and a
// single-member struct array memref<8x1xi64>. cgeist used to build a memref
// of memrefs for the array (memref<16xmemref<2xi64>>) and assert on the
// member access (CommonFieldLookup, machsuite bfs_bulk / bfs_queue, W8b), and
// a struct assignment between two such rows asserted on the rank-2
// reference (`out[i] = p`, `out[1][i] = pos[2][i]`); it is copied member by
// member.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <stdint.h>

typedef struct { uint64_t dst; } edge_t;
typedef struct { uint64_t edge_begin; uint64_t edge_end; } node_t;
typedef struct { double x, y, z; } dvec;

// CHECK-LABEL: func.func @bfs(
// CHECK-SAME:    %[[N:[^:]*]]: memref<16x2xi64>, %[[E:[^:]*]]: memref<64x1xi64>, %[[L:[^:]*]]: memref<16xi8>
// CHECK:         affine.for %[[I:.*]] = 0 to 16 {
// CHECK:           %[[B:.*]] = affine.load %[[N]][%[[I]], 0] : memref<16x2xi64>
// CHECK:           %[[END:.*]] = affine.load %[[N]][%[[I]], 1] : memref<16x2xi64>
// CHECK:           memref.load %[[E]][%{{.*}}, %{{.*}}] : memref<64x1xi64>
void bfs(node_t nodes[16], edge_t edges[64], int8_t level[16]) {
  for (int n = 0; n < 16; n++) {
    uint64_t e = nodes[n].edge_begin;
    uint64_t end = nodes[n].edge_end;
    for (int i = 0; i < 4; i++) {
      if (e < end) {
        level[edges[e].dst & 15] = 1;
        e++;
      }
    }
  }
}

// CHECK-LABEL: func.func @single(
// CHECK-SAME:    %[[E:[^:]*]]: memref<8x1xi64>, %[[O:[^:]*]]: memref<8xi64>
// CHECK:         %[[V:.*]] = affine.load %[[E]][%[[I:.*]], 0] : memref<8x1xi64>
// CHECK:         affine.store %[[V]], %[[O]][%[[I]]] : memref<8xi64>
void single(edge_t e[8], uint64_t out[8]) {
  for (int i = 0; i < 8; i++)
    out[i] = e[i].dst;
}

// CHECK-LABEL: func.func @copy_local(
// CHECK-SAME:    %[[P:[^:]*]]: memref<4x4x8x3xf64>, %[[O:[^:]*]]: memref<8x3xf64>
// CHECK:         %[[X:.*]] = affine.load %[[P]][1, 2, %[[I:.*]], 0] : memref<4x4x8x3xf64>
// CHECK:         %[[Y:.*]] = affine.load %[[P]][1, 2, %[[I]], 1] : memref<4x4x8x3xf64>
// CHECK:         %[[Z:.*]] = affine.load %[[P]][1, 2, %[[I]], 2] : memref<4x4x8x3xf64>
// CHECK:         affine.store %[[X]], %[[O]][%[[I]], 0] : memref<8x3xf64>
// CHECK:         affine.store %[[Y]], %[[O]][%[[I]], 1] : memref<8x3xf64>
// CHECK:         affine.store %[[Z]], %[[O]][%[[I]], 2] : memref<8x3xf64>
void copy_local(dvec pos[4][4][8], dvec out[8]) {
  for (int i = 0; i < 8; i++) {
    dvec p = pos[1][2][i];
    out[i] = p;
  }
}

// CHECK-LABEL: func.func @copy_rows(
// CHECK-SAME:    %[[P:[^:]*]]: memref<4x8x3xf64>, %[[O:[^:]*]]: memref<4x8x3xf64>
// CHECK:         %[[X:.*]] = affine.load %[[P]][2, %[[I:.*]], 0] : memref<4x8x3xf64>
// CHECK:         affine.store %[[X]], %[[O]][1, %[[I]], 0] : memref<4x8x3xf64>
// CHECK:         %[[Y:.*]] = affine.load %[[P]][2, %[[I]], 1] : memref<4x8x3xf64>
// CHECK:         affine.store %[[Y]], %[[O]][1, %[[I]], 1] : memref<4x8x3xf64>
// CHECK:         %[[Z:.*]] = affine.load %[[P]][2, %[[I]], 2] : memref<4x8x3xf64>
// CHECK:         affine.store %[[Z]], %[[O]][1, %[[I]], 2] : memref<4x8x3xf64>
void copy_rows(dvec pos[4][8], dvec out[4][8]) {
  for (int i = 0; i < 8; i++)
    out[1][i] = pos[2][i];
}
