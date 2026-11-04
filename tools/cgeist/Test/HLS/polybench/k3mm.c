// PolyBench 3mm from HLSFactory (hls_dataset_sources/polybench/k3mm),
// with one configuration of its opt_template.tcl directives written as
// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the
// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).
// Generated; the CHECK lines follow from the directives alone.
// RUN: cgeist %s --function=k3mm %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: cgeist %s --function=k3mm %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT-NOT: warning
// REPORT-NOT: not applied
// REPORT-NOT: rejected
#define N 64
#define DATA_TYPE float

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


void k3mm(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N], DATA_TYPE E_out[N][N])
{
    int i, j, k;

    DATA_TYPE buff_A[N][N];
    DATA_TYPE buff_B[N][N];
    DATA_TYPE buff_C[N][N];
    DATA_TYPE buff_D[N][N];
    DATA_TYPE buff_E_out[N][N];
    DATA_TYPE tmp1[N][N];
    DATA_TYPE tmp2[N][N];

#pragma HLS bind_storage variable=A type=ram_1p
#pragma HLS bind_storage variable=B type=ram_1p
#pragma HLS bind_storage variable=C type=ram_1p
#pragma HLS bind_storage variable=D type=ram_1p
#pragma HLS interface ap_fifo port=E_out
#pragma HLS array_partition variable=A cyclic factor=4 dim=2
#pragma HLS array_partition variable=B cyclic factor=4 dim=2
#pragma HLS array_partition variable=C cyclic factor=4 dim=2
#pragma HLS array_partition variable=D cyclic factor=4 dim=2
#pragma HLS array_partition variable=E_out cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_B cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_C cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_D cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_E_out cyclic factor=4 dim=2
#pragma HLS array_partition variable=tmp1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=tmp2 cyclic factor=4 dim=2
    lprd_1: for (i = 0; i < N; i++) {
        lprd_2: for (j = 0; j < N; j++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
            buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            buff_C[i][j] = C[i][j];
            buff_D[i][j] = D[i][j];
            buff_E_out[i][j] = 0;
            tmp1[i][j] = 0;
            tmp2[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
            lp3: for (k = 0; k < N; k++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
                tmp1[i][j] += buff_A[i][k] * buff_B[k][j];
            }
        }
    }

    lp4: for (i = 0; i < N; i++) {
        lp5: for (j = 0; j < N; j++) {
            lp6: for (k = 0; k < N; k++) {
                tmp2[i][j] += buff_C[i][k] * buff_D[k][j];
            }
        }
    }
    
    lp7: for (i = 0; i < N; i++) {
        lp8: for (j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
            lp9: for (k = 0; k < N; k++) {
                buff_E_out[i][j] += tmp1[i][k] * tmp2[k][j];
            }
        }
    }

    lpwr_1: for (i = 0; i < N; i++) {
        lpwr_2: for (j = 0; j < N; j++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
            E_out[i][j] = buff_E_out[i][j];
        }
    }
}

// REPORT-DAG: k3mm.c:34:1: remark: '#pragma HLS bind_storage variable=A type=ram_1p' -> argument #0 (variable 'A') of @k3mm
// REPORT-DAG: k3mm.c:35:1: remark: '#pragma HLS bind_storage variable=B type=ram_1p' -> argument #1 (variable 'B') of @k3mm
// REPORT-DAG: k3mm.c:36:1: remark: '#pragma HLS bind_storage variable=C type=ram_1p' -> argument #2 (variable 'C') of @k3mm
// REPORT-DAG: k3mm.c:37:1: remark: '#pragma HLS bind_storage variable=D type=ram_1p' -> argument #3 (variable 'D') of @k3mm
// REPORT-DAG: k3mm.c:38:1: remark: '#pragma HLS interface ap_fifo port=E_out' -> argument #4 (variable 'E_out') of @k3mm
// REPORT-DAG: k3mm.c:39:1: remark: '#pragma HLS array_partition variable=A cyclic factor=4 dim=2' -> argument #0 (variable 'A') of @k3mm
// REPORT-DAG: k3mm.c:40:1: remark: '#pragma HLS array_partition variable=B cyclic factor=4 dim=2' -> argument #1 (variable 'B') of @k3mm
// REPORT-DAG: k3mm.c:41:1: remark: '#pragma HLS array_partition variable=C cyclic factor=4 dim=2' -> argument #2 (variable 'C') of @k3mm
// REPORT-DAG: k3mm.c:42:1: remark: '#pragma HLS array_partition variable=D cyclic factor=4 dim=2' -> argument #3 (variable 'D') of @k3mm
// REPORT-DAG: k3mm.c:43:1: remark: '#pragma HLS array_partition variable=E_out cyclic factor=4 dim=2' -> argument #4 (variable 'E_out') of @k3mm
// REPORT-DAG: k3mm.c:44:1: remark: '#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A') in @k3mm
// REPORT-DAG: k3mm.c:45:1: remark: '#pragma HLS array_partition variable=buff_B cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_B') in @k3mm
// REPORT-DAG: k3mm.c:46:1: remark: '#pragma HLS array_partition variable=buff_C cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_C') in @k3mm
// REPORT-DAG: k3mm.c:47:1: remark: '#pragma HLS array_partition variable=buff_D cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_D') in @k3mm
// REPORT-DAG: k3mm.c:48:1: remark: '#pragma HLS array_partition variable=buff_E_out cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_E_out') in @k3mm
// REPORT-DAG: k3mm.c:49:1: remark: '#pragma HLS array_partition variable=tmp1 cyclic factor=4 dim=2' -> memref.alloca (variable 'tmp1') in @k3mm
// REPORT-DAG: k3mm.c:50:1: remark: '#pragma HLS array_partition variable=tmp2 cyclic factor=4 dim=2' -> memref.alloca (variable 'tmp2') in @k3mm
// REPORT-DAG: k3mm.c:53:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lprd_2') in @k3mm
// REPORT-DAG: k3mm.c:54:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lprd_2') in @k3mm
// REPORT-DAG: k3mm.c:68:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp3') in @k3mm
// REPORT-DAG: k3mm.c:69:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp3') in @k3mm
// REPORT-DAG: k3mm.c:85:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp8') in @k3mm
// REPORT-DAG: k3mm.c:86:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp8') in @k3mm
// REPORT-DAG: k3mm.c:95:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lpwr_2') in @k3mm
// REPORT-DAG: k3mm.c:96:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lpwr_2') in @k3mm

// IR-LABEL: func.func @k3mm(
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "B"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "C"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "D"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "E_out"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_B"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_C"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_D"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_E_out"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "tmp1"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "tmp2"}]
