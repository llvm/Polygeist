// PolyBench syrk from HLSFactory (hls_dataset_sources/polybench/syrk),
// with one configuration of its opt_template.tcl directives written as
// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the
// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).
// Generated; the CHECK lines follow from the directives alone.
// RUN: cgeist %s --function=syrk %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: cgeist %s --function=syrk %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT-NOT: warning
// REPORT-NOT: not applied
// REPORT-NOT: rejected
#define N 64
#define DATA_TYPE float


void syrk(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C_out[N][N]) {
	int i, j, k;

	DATA_TYPE buff_A0[N][N], buff_A1[N][N];
	DATA_TYPE buff_B[N][N];
	DATA_TYPE buff_C_out[N][N];

#pragma HLS bind_storage variable=A type=ram_1p
#pragma HLS bind_storage variable=B type=ram_1p
#pragma HLS interface ap_fifo port=C_out
#pragma HLS array_partition variable=A cyclic factor=4 dim=2
#pragma HLS array_partition variable=B cyclic factor=4 dim=2
#pragma HLS array_partition variable=C_out cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A0 cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_B cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_C_out cyclic factor=4 dim=2
	lprd_1: for (i = 0; i < N; i++) {
		lprd_2: for (j = 0; j < N; j++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
			buff_A0[i][j] = A[i][j];
			buff_A1[i][j] = A[i][j];
			buff_B[i][j] = B[i][j];
			buff_C_out[i][j] = 0;
		}
	}

	lp1: for (i = 0; i < N; i++) {
		lp2: for (j = 0; j < N; j++) {
			// if (j > i) continue;
			lp3: for (k = 0; k < N; k++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
				buff_C_out[i][j] += alpha * buff_A0[i][k] * buff_A1[j][k];
			}
		}
	}

	lp4: for (i = 0; i < N; i++) {
		lp5: for (j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
			// if (j > i) continue;
			buff_C_out[i][j] += beta * buff_B[i][j];
		}
	}

	

	lpwr_1: for(i = 0; i < N; i++) {
		lpwr_2: for(j = 0; j < N; j++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
			if (j > i) C_out[i][j] = 0;
			else C_out[i][j] = buff_C_out[i][j];
		}
	}
}


// REPORT-DAG: syrk.c:24:1: remark: '#pragma HLS bind_storage variable=A type=ram_1p' -> argument #2 (variable 'A') of @syrk
// REPORT-DAG: syrk.c:25:1: remark: '#pragma HLS bind_storage variable=B type=ram_1p' -> argument #3 (variable 'B') of @syrk
// REPORT-DAG: syrk.c:26:1: remark: '#pragma HLS interface ap_fifo port=C_out' -> argument #4 (variable 'C_out') of @syrk
// REPORT-DAG: syrk.c:27:1: remark: '#pragma HLS array_partition variable=A cyclic factor=4 dim=2' -> argument #2 (variable 'A') of @syrk
// REPORT-DAG: syrk.c:28:1: remark: '#pragma HLS array_partition variable=B cyclic factor=4 dim=2' -> argument #3 (variable 'B') of @syrk
// REPORT-DAG: syrk.c:29:1: remark: '#pragma HLS array_partition variable=C_out cyclic factor=4 dim=2' -> argument #4 (variable 'C_out') of @syrk
// REPORT-DAG: syrk.c:30:1: remark: '#pragma HLS array_partition variable=buff_A0 cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A0') in @syrk
// REPORT-DAG: syrk.c:31:1: remark: '#pragma HLS array_partition variable=buff_A1 cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A1') in @syrk
// REPORT-DAG: syrk.c:32:1: remark: '#pragma HLS array_partition variable=buff_B cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_B') in @syrk
// REPORT-DAG: syrk.c:33:1: remark: '#pragma HLS array_partition variable=buff_C_out cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_C_out') in @syrk
// REPORT-DAG: syrk.c:36:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lprd_2') in @syrk
// REPORT-DAG: syrk.c:37:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lprd_2') in @syrk
// REPORT-DAG: syrk.c:49:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp3') in @syrk
// REPORT-DAG: syrk.c:50:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp3') in @syrk
// REPORT-DAG: syrk.c:58:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp5') in @syrk
// REPORT-DAG: syrk.c:59:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp5') in @syrk
// REPORT-DAG: syrk.c:69:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lpwr_2') in @syrk
// REPORT-DAG: syrk.c:70:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lpwr_2') in @syrk

// IR-LABEL: func.func @syrk(
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "B"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "C_out"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A0"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A1"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_B"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_C_out"}]
