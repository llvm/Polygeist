// PolyBench mvt from HLSFactory (hls_dataset_sources/polybench/mvt),
// with one configuration of its opt_template.tcl directives written as
// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the
// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).
// Generated; the CHECK lines follow from the directives alone.
// RUN: cgeist %s --function=mvt %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: cgeist %s --function=mvt %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT-NOT: warning
// REPORT-NOT: not applied
// REPORT-NOT: rejected
#define N 64
#define DATA_TYPE float


void mvt(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N])
{
	int i, j;
	DATA_TYPE buff_A[N][N];
	DATA_TYPE buff_x1[N];
	DATA_TYPE buff_x2[N];
	DATA_TYPE buff_y1[N];
	DATA_TYPE buff_y2[N];

#pragma HLS bind_storage variable=A type=ram_1p
#pragma HLS bind_storage variable=x1 type=ram_1p
#pragma HLS bind_storage variable=x2 type=ram_1p
#pragma HLS bind_storage variable=y1 type=ram_1p
#pragma HLS interface ap_fifo port=y2
#pragma HLS interface ap_fifo port=x1_out
#pragma HLS interface ap_fifo port=x2_out
#pragma HLS array_partition variable=A cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_x1 cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_x2 cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_y1 cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_y2 cyclic factor=4 dim=1
	lprd_1: for (i = 0; i < N; i++) {
		buff_x1[i] = x1[i];
		buff_x2[i] = x2[i];
		buff_y1[i] = y1[i];
		buff_y2[i] = y2[i];
    	lprd_2: for (j = 0; j < N; j++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
			buff_A[i][j] = A[i][j];
		}
	}

	lp1: for (i = 0; i < N; i++) {
    	lp2: for (j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
      		buff_x1[i] = buff_x1[i] + buff_A[i][j] * buff_y1[j];
		}
	}

	lp3: for (i = 0; i < N; i++) {
		lp4: for (j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
    		buff_x2[i] = buff_x2[i] + buff_A[j][i] * buff_y2[j];
		}
	}
	
	lpwr: for (i = 0; i < N; i++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
		x1_out[i] = buff_x1[i];
		x2_out[i] = buff_x2[i];
	}
}

// REPORT-DAG: mvt.c:26:1: remark: '#pragma HLS bind_storage variable=A type=ram_1p' -> argument #0 (variable 'A') of @mvt
// REPORT-DAG: mvt.c:27:1: remark: '#pragma HLS bind_storage variable=x1 type=ram_1p' -> argument #1 (variable 'x1') of @mvt
// REPORT-DAG: mvt.c:28:1: remark: '#pragma HLS bind_storage variable=x2 type=ram_1p' -> argument #2 (variable 'x2') of @mvt
// REPORT-DAG: mvt.c:29:1: remark: '#pragma HLS bind_storage variable=y1 type=ram_1p' -> argument #3 (variable 'y1') of @mvt
// REPORT-DAG: mvt.c:30:1: remark: '#pragma HLS interface ap_fifo port=y2' -> argument #4 (variable 'y2') of @mvt
// REPORT-DAG: mvt.c:31:1: remark: '#pragma HLS interface ap_fifo port=x1_out' -> argument #5 (variable 'x1_out') of @mvt
// REPORT-DAG: mvt.c:32:1: remark: '#pragma HLS interface ap_fifo port=x2_out' -> argument #6 (variable 'x2_out') of @mvt
// REPORT-DAG: mvt.c:33:1: remark: '#pragma HLS array_partition variable=A cyclic factor=4 dim=2' -> argument #0 (variable 'A') of @mvt
// REPORT-DAG: mvt.c:34:1: remark: '#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A') in @mvt
// REPORT-DAG: mvt.c:35:1: remark: '#pragma HLS array_partition variable=buff_x1 cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_x1') in @mvt
// REPORT-DAG: mvt.c:36:1: remark: '#pragma HLS array_partition variable=buff_x2 cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_x2') in @mvt
// REPORT-DAG: mvt.c:37:1: remark: '#pragma HLS array_partition variable=buff_y1 cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_y1') in @mvt
// REPORT-DAG: mvt.c:38:1: remark: '#pragma HLS array_partition variable=buff_y2 cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_y2') in @mvt
// REPORT-DAG: mvt.c:45:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lprd_2') in @mvt
// REPORT-DAG: mvt.c:46:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lprd_2') in @mvt
// REPORT-DAG: mvt.c:53:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp2') in @mvt
// REPORT-DAG: mvt.c:54:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp2') in @mvt
// REPORT-DAG: mvt.c:61:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp4') in @mvt
// REPORT-DAG: mvt.c:62:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp4') in @mvt
// REPORT-DAG: mvt.c:68:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lpwr') in @mvt
// REPORT-DAG: mvt.c:69:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lpwr') in @mvt

// IR-LABEL: func.func @mvt(
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_x1"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_x2"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_y1"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_y2"}]
