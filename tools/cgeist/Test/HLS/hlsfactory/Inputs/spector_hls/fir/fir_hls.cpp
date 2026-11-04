#include "fir_hls.h"

void fir_hls(
    float dataPtr[INPUT_SIZE],
    float filterPtr[FILTER_SIZE],
    float resultPtr[RESULT_SIZE],
    unsigned int totalInputLength,
    const int paddedSingleInputLength,
    const int totalFiltersLength) {
    PRAGMA_HLS(
        HLS ARRAY_PARTITION variable = dataPtr cyclic factor =
            KNOB_NUM_PARALLEL)
    PRAGMA_HLS(
        HLS ARRAY_PARTITION variable = filterPtr cyclic factor =
            KNOB_NUM_PARALLEL)
    PRAGMA_HLS(
        HLS ARRAY_PARTITION variable = resultPtr cyclic factor =
            KNOB_NUM_PARALLEL)

    for (int tid = 0; tid < KNOB_NUM_PARALLEL; tid++) {
        // Sliding windows
        float ai_a0[FILTER_LENGTH + KNOB_NUM_PARALLEL - 1];
        float ai_b0[FILTER_LENGTH + KNOB_NUM_PARALLEL - 1];

        // Filter coefficients
        float coef_real[FILTER_LENGTH];
        float coef_imag[FILTER_LENGTH];
        PRAGMA_HLS(
            HLS ARRAY_PARTITION variable = coef_real cyclic factor =
                KNOB_COEF_SHIFT)
        PRAGMA_HLS(
            HLS ARRAY_PARTITION variable = coef_imag cyclic factor =
                KNOB_COEF_SHIFT)

        int ilen, k;

        // Initialize sliding windows
        //#pragma HLS UNROLL
        for (ilen = 0; ilen < FILTER_LENGTH + KNOB_NUM_PARALLEL - 1; ilen++) {
            ai_a0[ilen] = 0.0f;
            ai_b0[ilen] = 0.0f;
        }

        uchar load_filter = 1;
        ushort load_filter_index = tid * totalFiltersLength;
        uchar num_coefs_loaded = 0;
        ushort ifilter = 0;

        // *********
        // Main loop
        // *********
        //    #pragma unroll KNOB_UNROLL_TOTAL
        for (ilen = 0; ilen < NUMITERATIONSKERNELARG; ilen++) {
            int currentIdx =
                2 * tid * totalInputLength + 2 * KNOB_NUM_PARALLEL * ilen;

            // Local filter results
            float firReal[KNOB_NUM_PARALLEL] = {0.0f};
            float firImag[KNOB_NUM_PARALLEL] = {0.0f};

            // Shift sliding windows
            for (k = 0; k < FILTER_LENGTH - 1; k++) {
                PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_FILTER_1)
                ai_a0[k] = ai_a0[k + KNOB_NUM_PARALLEL];
                ai_b0[k] = ai_b0[k + KNOB_NUM_PARALLEL];
            }

            // Shift in complex data point(s) to process

            for (k = 0; k < KNOB_NUM_PARALLEL; k++) {
                // #pragma HLS UNROLL
                int dataIdx = currentIdx + 2 * k;
                ai_a0[k + FILTER_LENGTH - 1] = dataPtr[dataIdx];
                ai_b0[k + FILTER_LENGTH - 1] = dataPtr[dataIdx + 1];
            }

            // Also shift in the filter coefficients for every set of data to
            // process
            //
            // Shift the cofficients in 8 complex points every clock cycle
            // It will take 16 clock cycles to shift all 128 coefficients in.
            // Thus, we need to pad the incoming data with 16 complex points of
            // 0 at the beginning of every new dataset to ensure data is
            // aligned.
            //
            // Note: We parameterize the number cited above.
            //
            if (load_filter) {

                // Shift coefficients
                for (k = 0; k < FILTER_LENGTH - KNOB_COEF_SHIFT;
                     k += KNOB_COEF_SHIFT) {
                    PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_FILTER_2)
                    for (int i = 0; i < KNOB_COEF_SHIFT; i++) {
                        PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_COEF_SHIFT_1)
                        coef_real[k + i] = coef_real[k + i + KNOB_COEF_SHIFT];
                        coef_imag[k + i] = coef_imag[k + i + KNOB_COEF_SHIFT];
                    }
                }

                // Load new coefficients
                for (int i = 0; i < KNOB_COEF_SHIFT; i++) {
                    PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_COEF_SHIFT_2)
                    coef_real[FILTER_LENGTH - (KNOB_COEF_SHIFT - i)] =
                        filterPtr[2 * load_filter_index + 2 * i];
                    coef_imag[FILTER_LENGTH - (KNOB_COEF_SHIFT - i)] =
                        filterPtr[2 * load_filter_index + 2 * i + 1];
                }

                load_filter_index += KNOB_COEF_SHIFT;

                if (++num_coefs_loaded == NUM_COEF_LOADS) {
                    load_filter = 0;
                    num_coefs_loaded = 0;
                }
            }

            // This is the core computation of the FIR filter

            for (k = FILTER_LENGTH - 1; k >= 0; k--) {
                PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_PARALLEL_2)
                for (int i = 0; i < KNOB_NUM_PARALLEL; i++) {
                    PRAGMA_HLS(HLS UNROLL factor = KNOB_NUM_PARALLEL)
                    firReal[i] +=
                        ai_a0[k + i] * coef_real[FILTER_LENGTH - 1 - k] -
                        ai_b0[k + i] * coef_imag[FILTER_LENGTH - 1 - k];
                    firImag[i] +=
                        ai_a0[k + i] * coef_imag[FILTER_LENGTH - 1 - k] +
                        ai_b0[k + i] * coef_real[FILTER_LENGTH - 1 - k];
                }
            }

            // Writing back the computational result
            for (int i = 0; i < KNOB_NUM_PARALLEL; i++) {
                PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_PARALLEL_3)
                int resultIdx = currentIdx + 2 * i;

                resultPtr[resultIdx] = firReal[i];
                resultPtr[resultIdx + 1] = firImag[i];
            }

            // The ifilter variable is a counter that counts up to the number of
            // data inputs per filter to process.  When it reaches
            // paddedSingleInputLength, we will know that it is time to load in
            // new filter coefficients for the next batch of data, and reset the
            // counter.
            if (ifilter == paddedSingleInputLength) {
                load_filter = 1;
            }
            if (ifilter == paddedSingleInputLength) {
                ifilter = 0;
            } else
                ifilter += KNOB_NUM_PARALLEL;
        }
    }
}

void fir_fpga(
    float dataPtr[INPUT_SIZE],
    float filterPtr[FILTER_SIZE],
    float resultPtr[RESULT_SIZE],
    unsigned int totalInputLength,
    const int paddedSingleInputLength,
    const int totalFiltersLength,
    unsigned int tid) {

    // Sliding windows
    float ai_a0[FILTER_LENGTH + KNOB_NUM_PARALLEL - 1];
    float ai_b0[FILTER_LENGTH + KNOB_NUM_PARALLEL - 1];

    // Filter coefficients
    float coef_real[FILTER_LENGTH];
    float coef_imag[FILTER_LENGTH];

    int ilen, k;

    // Initialize sliding windows
    for (ilen = 0; ilen < FILTER_LENGTH + KNOB_NUM_PARALLEL - 1; ilen++) {
#pragma HLS UNROLL
        ai_a0[ilen] = 0.0f;
        ai_b0[ilen] = 0.0f;
    }

    uchar load_filter = 1;
    ushort load_filter_index = tid * totalFiltersLength;
    uchar num_coefs_loaded = 0;
    ushort ifilter = 0;

    // *********
    // Main loop
    // *********
    //    #pragma unroll KNOB_UNROLL_TOTAL
    for (ilen = 0; ilen < NUMITERATIONSKERNELARG; ilen++) {
        int currentIdx =
            2 * tid * totalInputLength + 2 * KNOB_NUM_PARALLEL * ilen;

        // Local filter results
        float firReal[KNOB_NUM_PARALLEL] = {0.0f};
        float firImag[KNOB_NUM_PARALLEL] = {0.0f};

        // Shift sliding windows
        for (k = 0; k < FILTER_LENGTH - 1; k++) {
            PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_FILTER_1)
            ai_a0[k] = ai_a0[k + KNOB_NUM_PARALLEL];
            ai_b0[k] = ai_b0[k + KNOB_NUM_PARALLEL];
        }

        // Shift in complex data point(s) to process
        for (k = 0; k < KNOB_NUM_PARALLEL; k++) {
            PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_PARALLEL_1)
            int dataIdx = currentIdx + 2 * k;
            ai_a0[k + FILTER_LENGTH - 1] = dataPtr[dataIdx];
            ai_b0[k + FILTER_LENGTH - 1] = dataPtr[dataIdx + 1];
        }

        // Also shift in the filter coefficients for every set of data to
        // process
        //
        // Shift the cofficients in 8 complex points every clock cycle
        // It will take 16 clock cycles to shift all 128 coefficients in.
        // Thus, we need to pad the incoming data with 16 complex points of 0
        // at the beginning of every new dataset to ensure data is aligned.
        //
        // Note: We parameterize the number cited above.
        //
        if (load_filter) {

            // Shift coefficients
            for (k = 0; k < FILTER_LENGTH - KNOB_COEF_SHIFT;
                 k += KNOB_COEF_SHIFT) {
                PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_FILTER_2)

                for (int i = 0; i < KNOB_COEF_SHIFT; i++) {
                    PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_COEF_SHIFT_1)
                    coef_real[k + i] = coef_real[k + i + KNOB_COEF_SHIFT];
                    coef_imag[k + i] = coef_imag[k + i + KNOB_COEF_SHIFT];
                }
            }

            // Load new coefficients
            for (int i = 0; i < KNOB_COEF_SHIFT; i++) {
                PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_COEF_SHIFT_2)
                coef_real[FILTER_LENGTH - (KNOB_COEF_SHIFT - i)] =
                    filterPtr[2 * load_filter_index + 2 * i];
                coef_imag[FILTER_LENGTH - (KNOB_COEF_SHIFT - i)] =
                    filterPtr[2 * load_filter_index + 2 * i + 1];
            }

            load_filter_index += KNOB_COEF_SHIFT;

            if (++num_coefs_loaded == NUM_COEF_LOADS) {
                load_filter = 0;
                num_coefs_loaded = 0;
            }
        }

        // This is the core computation of the FIR filter
        for (k = FILTER_LENGTH - 1; k >= 0; k--) {
            PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_FILTER_3)
            for (int i = 0; i < KNOB_NUM_PARALLEL; i++) {
                PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_PARALLEL_2)
                firReal[i] += ai_a0[k + i] * coef_real[FILTER_LENGTH - 1 - k] -
                              ai_b0[k + i] * coef_imag[FILTER_LENGTH - 1 - k];
                firImag[i] += ai_a0[k + i] * coef_imag[FILTER_LENGTH - 1 - k] +
                              ai_b0[k + i] * coef_real[FILTER_LENGTH - 1 - k];
            }
        }

        // Writing back the computational result
        for (int i = 0; i < KNOB_NUM_PARALLEL; i++) {
            PRAGMA_HLS(HLS UNROLL factor = KNOB_UNROLL_PARALLEL_3)
            int resultIdx = currentIdx + 2 * i;

            resultPtr[resultIdx] = firReal[i];
            resultPtr[resultIdx + 1] = firImag[i];
        }

        // The ifilter variable is a counter that counts up to the number of
        // data inputs per filter to process.  When it reaches
        // paddedSingleInputLength, we will know that it is time to load in new
        // filter coefficients for the next batch of data, and reset the
        // counter.
        if (ifilter == paddedSingleInputLength) {
            load_filter = 1;
        }
        if (ifilter == paddedSingleInputLength) {
            ifilter = 0;
        } else
            ifilter += KNOB_NUM_PARALLEL;
    }
    /*
      Stop the timer.  Print out the
      total time in Seconds it took to do the TDFIR.
    */
    // printf("Done.\n  Latency: %f s.\n", tdFirVars->time.data[0])
    // printf("  Throughput: %.3f GFLOPs.\n", 268.44f / tdFirVars->time.data[0]
    // / 1000.0f );
}
