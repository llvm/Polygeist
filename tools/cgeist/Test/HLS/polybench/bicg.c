// PolyBench bicg from HLSFactory (hls_dataset_sources/polybench/bicg),
// with one configuration of its opt_template.tcl directives written as
// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the
// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).
// Generated; the CHECK lines follow from the directives alone.
// RUN: cgeist %s --function=bicg %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: cgeist %s --function=bicg %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// REPORT-NOT: warning
// REPORT-NOT: not applied
// REPORT-NOT: rejected
#define N 64
#define DATA_TYPE float


void bicg(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N])
{
    int i, j;

	DATA_TYPE buff_A[N][N];
	DATA_TYPE buff_p[N];
	DATA_TYPE buff_r[N];
	DATA_TYPE buff_s_out[N];
	DATA_TYPE buff_q_out[N];

#pragma HLS bind_storage variable=A type=ram_1p
#pragma HLS bind_storage variable=r type=ram_1p
#pragma HLS bind_storage variable=p type=ram_1p
#pragma HLS interface ap_fifo port=s_out
#pragma HLS interface ap_fifo port=q_out
#pragma HLS array_partition variable=A cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2
#pragma HLS array_partition variable=buff_p cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_r cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_s_out cyclic factor=4 dim=1
#pragma HLS array_partition variable=buff_q_out cyclic factor=4 dim=1
	lprd_1: for (i = 0; i < N; i++) {
		buff_p[i] = p[i];
		buff_r[i] = r[i];
		buff_s_out[i] = 0;
		buff_q_out[i] = 0;
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
	        buff_s_out[j] = buff_s_out[j] + buff_A[i][j] * buff_r[i];
		}
	}

	lp3: for (i = 0; i < N; i++) {
        lp4: for (j = 0; j < N; j++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=2
	        buff_q_out[i] = buff_q_out[i] + buff_A[i][j] * buff_p[j];
	    }
    }

	lpwr: for (i = 0; i < N; i++) {
#pragma HLS unroll factor=4
#pragma HLS pipeline
		s_out[i] = buff_s_out[i];
		q_out[i] = buff_q_out[i];
	}
}

// REPORT-DAG: bicg.c:27:1: remark: '#pragma HLS bind_storage variable=A type=ram_1p' -> argument #0 (variable 'A') of @bicg
// REPORT-DAG: bicg.c:28:1: remark: '#pragma HLS bind_storage variable=r type=ram_1p' -> argument #2 (variable 'r') of @bicg
// REPORT-DAG: bicg.c:29:1: remark: '#pragma HLS bind_storage variable=p type=ram_1p' -> argument #1 (variable 'p') of @bicg
// REPORT-DAG: bicg.c:30:1: remark: '#pragma HLS interface ap_fifo port=s_out' -> argument #3 (variable 's_out') of @bicg
// REPORT-DAG: bicg.c:31:1: remark: '#pragma HLS interface ap_fifo port=q_out' -> argument #4 (variable 'q_out') of @bicg
// REPORT-DAG: bicg.c:32:1: remark: '#pragma HLS array_partition variable=A cyclic factor=4 dim=2' -> argument #0 (variable 'A') of @bicg
// REPORT-DAG: bicg.c:33:1: remark: '#pragma HLS array_partition variable=buff_A cyclic factor=4 dim=2' -> memref.alloca (variable 'buff_A') in @bicg
// REPORT-DAG: bicg.c:34:1: remark: '#pragma HLS array_partition variable=buff_p cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_p') in @bicg
// REPORT-DAG: bicg.c:35:1: remark: '#pragma HLS array_partition variable=buff_r cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_r') in @bicg
// REPORT-DAG: bicg.c:36:1: remark: '#pragma HLS array_partition variable=buff_s_out cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_s_out') in @bicg
// REPORT-DAG: bicg.c:37:1: remark: '#pragma HLS array_partition variable=buff_q_out cyclic factor=4 dim=1' -> memref.alloca (variable 'buff_q_out') in @bicg
// REPORT-DAG: bicg.c:44:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lprd_2') in @bicg
// REPORT-DAG: bicg.c:45:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lprd_2') in @bicg
// REPORT-DAG: bicg.c:52:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp2') in @bicg
// REPORT-DAG: bicg.c:53:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp2') in @bicg
// REPORT-DAG: bicg.c:60:1: remark: '#pragma HLS pipeline II=1' -> {{affine|scf}}.for (loop 'lp4') in @bicg
// REPORT-DAG: bicg.c:61:1: remark: '#pragma HLS unroll factor=2' -> {{affine|scf}}.for (loop 'lp4') in @bicg
// REPORT-DAG: bicg.c:67:1: remark: '#pragma HLS unroll factor=4' -> {{affine|scf}}.for (loop 'lpwr') in @bicg
// REPORT-DAG: bicg.c:68:1: remark: '#pragma HLS pipeline' -> {{affine|scf}}.for (loop 'lpwr') in @bicg

// IR-LABEL: func.func @bicg(
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "A"}]
// IR-DAG: hls.array_partition = [{dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_A"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_p"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_r"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_s_out"}]
// IR-DAG: hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buff_q_out"}]
