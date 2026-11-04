// PolyBench syr2k from HLSFactory (hls_dataset_sources/polybench/syr2k),
// with one configuration of its opt_template.tcl directives written as
// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the
// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).
// Generated; the CHECK lines follow from the directives alone.
// RUN: cgeist %s --function=syr2k %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: cgeist %s --function=syr2k %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// buff_B1 is written but never read in the original source, so it is
// removed (with a warning) instead of annotated.
// REPORT-NOT: rejected
#define N 64
#define DATA_TYPE float


void syr2k(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N]){
	int i, j, k;
	DATA_TYPE buff_A0[N][N], buff_A1[N][N];
	DATA_TYPE buff_B0[N][N], buff_B1[N][N];
	DATA_TYPE buff_C[N][N];
	DATA_TYPE buff_D_out[N][N];
	DATA_TYPE tmp1[N][N];
	DATA_TYPE tmp2[N][N];


#pragma HLS bind_storage variable=A type=ram_1p
#pragma HLS bind_storage variable=B type=ram_1p
#pragma HLS bind_storage variable=C type=ram_1p
#pragma HLS interface ap_fifo port=D_out
#pragma HLS array_partition variable=A cyclic factor=4 dim=2
#pragma HLS array_partition variable=B cyclic factor=4 dim=2
#pragma HLS array_partition variable=C cyclic factor=4 dim=2
#pragma HLS array_partition variable=D_out cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A0 cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_B0 cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_B1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_C cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_D_out cyclic factor=4 dim=2
#pragma HLS array_partition variable=tmp1 cyclic factor=4 dim=2
#pragma HLS array_partition variable=tmp2 cyclic factor=4 dim=2
	lprd_1: for (i = 0; i < N; i++){
		lprd_2: for (j = 0; j < N; j++){
#pragma HLS unroll factor=4
#pragma HLS pipeline
			buff_A0[i][j] = A[i][j];
			buff_A1[i][j] = A[i][j];
			buff_B0[i][j] = B[i][j];
			buff_B1[i][j] = B[i][j];
			buff_C[i][j] = C[i][j];
			buff_D_out[i][j] = 0;
			tmp1[i][j] = 0;
			tmp2[i][j] = 0;
		}
	}

	lp1: for (i = 0; i < N; i++){
		lp2: for (j = 0; j < N; j++){
			// if (j > i) continue;
			lp3: for (k = 0; k < N; k++){
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
				tmp1[i][j] += alpha * buff_A0[i][k] * buff_B0[j][k];
			}
		}
	}

	lp4: for (i = 0; i < N; i++){
		lp5: for (j = 0; j < N; j++){
			// if (j > i) continue;
			lp6: for (k = 0; k < N; k++){
				tmp2[i][j] += alpha * buff_B0[i][k] * buff_A1[j][k];
			}
		}
	}

	lp7: for (i = 0; i < N; i++){
		lp8: for (j = 0; j < N; j++){
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
			//if (j > i) continue;
			buff_D_out[i][j] = tmp1[i][j] + tmp2[i][j] + beta * buff_C[i][j];
		}
	}

	lpwr_1: for(i = 0; i < N; i++){
		lpwr_2: for(j = 0; j < N; j++){
#pragma HLS unroll factor=4
#pragma HLS pipeline
			if (j > i) D_out[i][j] = 0;
			else D_out[i][j] = buff_D_out[i][j];
		}
	}
}


// REPORT-DAG: syr2k.c:27:1: remark: '#pragma HLS bind_storage variable=A type=ram_1p' -> argument #2 (variable 'A') of @syr2k
// REPORT-DAG: syr2k.c:28:1: remark: '#pragma HLS bind_storage variable=B type=ram_1p' -> argument #3 (variable 'B') of @syr2k
// REPORT-DAG: syr2k.c:29:1: remark: '#pragma HLS bind_storage variable=C type=ram_1p' -> argument #4 (variable 'C') of @syr2k
// REPORT-DAG: syr2k.c:30:1: remark: '#pragma HLS interface ap_fifo port=D_out' -> argument #5 (variable 'D_out') of @syr2k
// REPORT-DAG: syr2k.c:31:1: remark: '#pragma HLS array_partition variable=A cyclic factor=4 dim=2' -> argument #2 (variable 'A') of @syr2k
// REPORT-DAG: syr2k.c:32:1: remark: '#pragma HLS array_partition variable=B cyclic factor=4 dim=2' -> argument #3 (variable 'B') of @syr2k
// REPORT-DAG: syr2k.c:33:1: remark: '#pragma HLS array_partition variable=C cyclic factor=4 dim=2' -> argument #4 (variable 'C') of @syr2k
// REPORT-DAG: syr2k.c:34:1: remark: '#pragma HLS array_partition variable=D_out cyclic factor=4 dim=2' -> argument #5 (variable 'D_out') of @syr2k
// REPORT-DAG: syr2k.c:35:1: remark: '#pragma HLS array_partition variable=buff_A0 cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A0') in @syr2k
// REPORT-DAG: syr2k.c:36:1: remark: '#pragma HLS array_partition variable=buff_A1 cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A1') in @syr2k
// REPORT-DAG: syr2k.c:37:1: remark: '#pragma HLS array_partition variable=buff_B0 cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_B0') in @syr2k
// REPORT-DAG: syr2k.c:38:1: remark: '#pragma HLS array_partition variable=buff_B1 cyclic factor=4 dim=2' -> not applied: the variable 'buff_B1' was removed during optimization
// REPORT-DAG: syr2k.c:39:1: remark: '#pragma HLS array_partition variable=buff_C cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_C') in @syr2k
// REPORT-DAG: syr2k.c:40:1: remark: '#pragma HLS array_partition variable=buff_D_out cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_D_out') in @syr2k
// REPORT-DAG: syr2k.c:41:1: remark: '#pragma HLS array_partition variable=tmp1 cyclic factor=4 dim=2' -> memref.alloca (variable 'tmp1') in @syr2k
// REPORT-DAG: syr2k.c:42:1: remark: '#pragma HLS array_partition variable=tmp2 cyclic factor=4 dim=2' -> memref.alloca (variable 'tmp2') in @syr2k
// REPORT-DAG: syr2k.c:45:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lprd_2') in @syr2k
// REPORT-DAG: syr2k.c:46:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lprd_2') in @syr2k
// REPORT-DAG: syr2k.c:62:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp3') in @syr2k
// REPORT-DAG: syr2k.c:63:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp3') in @syr2k
// REPORT-DAG: syr2k.c:80:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp8') in @syr2k
// REPORT-DAG: syr2k.c:81:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp8') in @syr2k
// REPORT-DAG: syr2k.c:89:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lpwr_2') in @syr2k
// REPORT-DAG: syr2k.c:90:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lpwr_2') in @syr2k

// IR-LABEL: func.func @syr2k(
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "B"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "C"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "D_out"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A0"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A1"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_B0"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_C"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_D_out"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "tmp1"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "tmp2"}]
