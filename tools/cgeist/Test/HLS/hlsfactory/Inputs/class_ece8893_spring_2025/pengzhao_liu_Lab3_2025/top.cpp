#include "dcl.h"
#include <hls_stream.h>

// We'll allow up to 64 nonzeros per column of B, which is feasible for 64x64.
#define MAX_NNZ_PER_COL 64

// Packet for one nonzero of A's row: (row, col, val, last)
struct ArowPkt_t {
    int    row;
    int    col;
    data_t val;
    bool   last; // end of this row
};

// Packet for one element of C's row: (row, col, val, last)
struct CrowPkt_t {
    int    row;
    int    col;
    data_t val;
    bool   last; // end of this row
};


//======================================================================
//  0)  loadB_Fully (sequential, no dataflow concurrency)
//      Reads B (CSC) from DRAM, caches in local arrays for random access
//======================================================================
static void loadB_Fully(
    data_t* values_B,    // M*K sized, but only col_ptr_B[M+1]-based actual nonzeros
    int*    row_indices_B,
    int*    col_ptr_B,

    data_t  bVals  [M][MAX_NNZ_PER_COL],
    int     bRowIdx[M][MAX_NNZ_PER_COL],
    int     bLen   [M]
)
{
    // 1) Copy col_ptr_B to local
    int local_colPtrB[M+1];
    for(int k=0; k<=M; k++){
    #pragma HLS PIPELINE II=1
        local_colPtrB[k] = col_ptr_B[k];
    }

    // 2) Init bLen
    for(int k=0; k<M; k++){
    #pragma HLS PIPELINE II=1
        bLen[k] = 0;
    }

    // 3) Read each column of B from DRAM into local arrays
    for(int k=0; k<M; k++){
        int startB = local_colPtrB[k];
        int endB   = local_colPtrB[k+1];
        int length = endB - startB;
        bLen[k]    = length;
        for(int i=0; i<length; i++){
        #pragma HLS PIPELINE II=1
            bVals  [k][i] = values_B    [startB + i];
            bRowIdx[k][i] = row_indices_B[startB + i];
        }
    }
}


//======================================================================
//  1) loadA_Stage
//     Reads A (CSR) row-by-row, streaming out nonzeros with a row-end marker
//======================================================================
static void loadA_Stage(
    data_t* values_A,
    int*    column_indices_A,
    int*    row_ptr_A,

    hls::stream<ArowPkt_t> &ArowStream
) {
#pragma HLS INLINE off

    // Copy row_ptr_A
    int local_rowPtrA[N+1];
    for(int i=0; i<=N; i++){
    #pragma HLS PIPELINE II=1
        local_rowPtrA[i] = row_ptr_A[i];
    }

    // For each row, stream out its nonzeros
    for(int i=0; i<N; i++){
        int startA = local_rowPtrA[i];
        int endA   = local_rowPtrA[i+1];
        for(int idxA=startA; idxA<endA; idxA++){
        #pragma HLS PIPELINE II=1
            ArowPkt_t pkt;
            pkt.row  = i;
            pkt.col  = column_indices_A[idxA];
            pkt.val  = values_A[idxA];
            pkt.last = false;
            ArowStream.write(pkt);
        }
        // End-of-row marker
        ArowPkt_t lastPkt;
        lastPkt.row = i; lastPkt.col=-1; lastPkt.val=0; lastPkt.last=true;
        ArowStream.write(lastPkt);
    }
}


//======================================================================
//  2) computeStage
//     Reads each row of A from the stream, multiplies by cached B in local arrays,
//     streams out row of C
//======================================================================
static void computeStage(
    hls::stream<ArowPkt_t> &ArowStream,
    const data_t bVals  [M][MAX_NNZ_PER_COL],
    const int    bRowIdx[M][MAX_NNZ_PER_COL],
    const int    bLen   [M],

    hls::stream<CrowPkt_t> &CrowStream
) {
#pragma HLS INLINE off

    static data_t rowAccum[K];
    #pragma HLS bind_storage variable=rowAccum type=ram_2p impl=bram

RowLoop: for(int i=0; i<N; i++){
        // init accumulators
        for(int j=0; j<K; j++){
        #pragma HLS PIPELINE II=1
            rowAccum[j] = 0;
        }

        // read all nonzeros in row i
        while(true){
        #pragma HLS PIPELINE II=1
            ArowPkt_t apkt = ArowStream.read();
            if(apkt.last){
                // done with row i
                break;
            }
            int    col = apkt.col;  // index in B
            data_t aval= apkt.val;

            // accumulate: rowAccum[rowB] += aval * B(col, rowB)
            int lenBk = bLen[col];
            for(int idx=0; idx<lenBk; idx++){
            #pragma HLS PIPELINE II=1
                int    rowB = bRowIdx[col][idx];
                data_t vb   = bVals   [col][idx];
                rowAccum[rowB] += (aval * vb);
            }
        }

        // output row i of C
        for(int j=0; j<K; j++){
        #pragma HLS PIPELINE II=1
            CrowPkt_t cp;
            cp.row  = i;
            cp.col  = j;
            cp.val  = rowAccum[j];
            cp.last = (j == (K-1));
            CrowStream.write(cp);
        }
    }
}


//======================================================================
//  3) storeStage
//     Consumes row-by-row C from CrowStream, writes to C[][] in DRAM
//======================================================================
static void storeStage(
    hls::stream<CrowPkt_t> &CrowStream,
    data_t C[N][K]
) {
#pragma HLS INLINE off

    // We get N*K packets total (row i, col j, val)
    // row i has K columns
    for(int i=0; i<N; i++){
        for(int j=0; j<K; j++){
        #pragma HLS PIPELINE II=1
            CrowPkt_t cp = CrowStream.read();
            C[cp.row][cp.col] = cp.val;
        }
    }
}


//======================================================================
//  Top-level: Two-phase approach
//     Phase 1: loadB_Fully (no dataflow concurrency)
//     Phase 2: dataflow pipeline of (loadA_Stage -> computeStage -> storeStage)
//======================================================================
void sparse_matrix_multiply_HLS(
    data_t values_A[N*M],
    int    column_indices_A[N*M],
    int    row_ptr_A[N+1],

    data_t values_B[M*K],
    int    row_indices_B[M*K],
    int    col_ptr_B[M+1],

    data_t C[N][K]
) {
#pragma HLS INTERFACE m_axi port=values_A         offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=row_ptr_A        offset=slave bundle=mem1

#pragma HLS INTERFACE m_axi port=values_B         offset=slave bundle=mem2
#pragma HLS INTERFACE m_axi port=row_indices_B    offset=slave bundle=mem2
#pragma HLS INTERFACE m_axi port=col_ptr_B        offset=slave bundle=mem2

#pragma HLS INTERFACE m_axi port=C                offset=slave bundle=mem3
#pragma HLS INTERFACE s_axilite port=return

    // Local arrays for B
    static data_t bVals  [M][MAX_NNZ_PER_COL];
    static int    bRowIdx[M][MAX_NNZ_PER_COL];
    static int    bLen   [M];
    #pragma HLS bind_storage variable=bVals   type=ram_1p impl=bram
    #pragma HLS bind_storage variable=bRowIdx type=ram_1p impl=bram
    #pragma HLS array_partition variable=bLen complete

    // PHASE 1: sequentially load B fully (no dataflow concurrency)
    loadB_Fully(values_B, row_indices_B, col_ptr_B, bVals, bRowIdx, bLen);

    // Streams for row-by-row pipeline of A -> compute -> C
    static hls::stream<ArowPkt_t> ArowStream("ArowStream");
    static hls::stream<CrowPkt_t> CrowStream("CrowStream");
    #pragma HLS STREAM variable=ArowStream depth=64
    #pragma HLS STREAM variable=CrowStream depth=64

    // PHASE 2: dataflow concurrency for A, compute, store
    #pragma HLS DATAFLOW
    loadA_Stage(values_A, column_indices_A, row_ptr_A, ArowStream);
    computeStage(ArowStream, bVals, bRowIdx, bLen, CrowStream);
    storeStage(CrowStream, C);
}
