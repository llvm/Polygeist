// PolyBench gesummv from HLSFactory (hls_dataset_sources/polybench/gesummv),
// with one configuration of its opt_template.tcl directives written as
// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the
// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).
// Generated; the CHECK lines follow from the directives alone.
// RUN: cgeist %s --function=gesummv %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: cgeist %s --function=gesummv %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT-NOT: warning
// REPORT-NOT: not applied
// REPORT-NOT: rejected
#define N 64
#define DATA_TYPE float


void gesummv(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
{
    int i, j;
	DATA_TYPE buff_A[N][N];
	DATA_TYPE buff_B[N][N];
	DATA_TYPE buff_x[N];
	DATA_TYPE buff_y_out[N];
	DATA_TYPE tmp1[N];
	DATA_TYPE tmp2[N];

#pragma HLS bind_storage variable=A type=ram_1p
#pragma HLS bind_storage variable=B type=ram_1p
#pragma HLS bind_storage variable=x type=ram_1p
#pragma HLS interface ap_fifo port=y_out
#pragma HLS array_partition variable=A cyclic factor=4 dim=2
#pragma HLS array_partition variable=B cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_B cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_x cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_y_out cyclic factor=4 dim=1
#pragma HLS array_partition variable=tmp1 cyclic factor=4 dim=1
#pragma HLS array_partition variable=tmp2 cyclic factor=4 dim=1
	lprd_1: for(i = 0; i < N; i++) {
		buff_x[i] = x[i];
		tmp1[i] = 0;
		tmp2[i] = 0;
		buff_y_out[i] = 0;
		lprd_2: for(j = 0; j < N; j++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
			buff_A[i][j] = A[i][j];
			buff_B[i][j] = B[i][j];
		}
	}

    lp1: for(i = 0; i < N; i++) {
        lp2: for(j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
	        tmp1[i] += alpha * buff_A[i][j] * buff_x[j];
        }
    }

	lp3: for(i = 0; i < N; i++) {
        lp4: for(j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
	        tmp2[i] += beta * buff_B[i][j] * buff_x[j];
        }
    }

	lp5: for(i = 0; i < N; i++) {
		buff_y_out[i] = tmp1[i] + tmp2[i];
	}

	lpwr: for(i = 0; i < N; i++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
		y_out[i] = buff_y_out[i];
	}
}

// REPORT-DAG: gesummv.c:27:1: remark: '#pragma HLS bind_storage variable=A type=ram_1p' -> argument #2 (variable 'A') of @gesummv
// REPORT-DAG: gesummv.c:28:1: remark: '#pragma HLS bind_storage variable=B type=ram_1p' -> argument #3 (variable 'B') of @gesummv
// REPORT-DAG: gesummv.c:29:1: remark: '#pragma HLS bind_storage variable=x type=ram_1p' -> argument #4 (variable 'x') of @gesummv
// REPORT-DAG: gesummv.c:30:1: remark: '#pragma HLS interface ap_fifo port=y_out' -> argument #5 (variable 'y_out') of @gesummv
// REPORT-DAG: gesummv.c:31:1: remark: '#pragma HLS array_partition variable=A cyclic factor=4 dim=2' -> argument #2 (variable 'A') of @gesummv
// REPORT-DAG: gesummv.c:32:1: remark: '#pragma HLS array_partition variable=B cyclic factor=4 dim=2' -> argument #3 (variable 'B') of @gesummv
// REPORT-DAG: gesummv.c:33:1: remark: '#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A') in @gesummv
// REPORT-DAG: gesummv.c:34:1: remark: '#pragma HLS array_partition variable=buff_B cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_B') in @gesummv
// REPORT-DAG: gesummv.c:35:1: remark: '#pragma HLS array_partition variable=buff_x cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_x') in @gesummv
// REPORT-DAG: gesummv.c:36:1: remark: '#pragma HLS array_partition variable=buff_y_out cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_y_out') in @gesummv
// REPORT-DAG: gesummv.c:37:1: remark: '#pragma HLS array_partition variable=tmp1 cyclic factor=4 dim=1' -> memref.alloca (variable 'tmp1') in @gesummv
// REPORT-DAG: gesummv.c:38:1: remark: '#pragma HLS array_partition variable=tmp2 cyclic factor=4 dim=1' -> memref.alloca (variable 'tmp2') in @gesummv
// REPORT-DAG: gesummv.c:45:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lprd_2') in @gesummv
// REPORT-DAG: gesummv.c:46:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lprd_2') in @gesummv
// REPORT-DAG: gesummv.c:54:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp2') in @gesummv
// REPORT-DAG: gesummv.c:55:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp2') in @gesummv
// REPORT-DAG: gesummv.c:62:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp4') in @gesummv
// REPORT-DAG: gesummv.c:63:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp4') in @gesummv
// REPORT-DAG: gesummv.c:73:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lpwr') in @gesummv
// REPORT-DAG: gesummv.c:74:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lpwr') in @gesummv

// IR-LABEL: func.func @gesummv(
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "B"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_B"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_x"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_y_out"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "tmp1"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "tmp2"}]
