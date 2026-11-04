#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#define CEIL(x) x % KNOB_NUM_PARALLEL == 0 ? x : x + 1
#define PADDINGENDTOTALLENGTH                                                  \
    (CEIL((FILTER_L - 1 + INPUT_L) / (KNOB_NUM_PARALLEL)) *                    \
         KNOB_NUM_PARALLEL -                                                   \
     INPUT_L)
#define PADDINGCOEFLOAD NUM_COEF_LOADS *KNOB_NUM_PARALLEL
#define PADDEDNUMINPUTPOINTS INPUT_L + PADDINGCOEFLOAD + PADDINGENDTOTALLENGTH
#define TOTALDATAINPUTLENGTHKERNELARG                                          \
    (PADDEDNUMINPUTPOINTS * (FILTER_NO / TOTAL_WORK_ITEMS))
#define NUMITERATIONSKERNELARG TOTALDATAINPUTLENGTHKERNELARG / KNOB_NUM_PARALLEL
#define PADDEDSINGLEINPUTLENGTHMINUS1KERNELARG                                 \
    PADDEDNUMINPUTPOINTS - KNOB_NUM_PARALLEL
#define EXTRAPADDING PADDINGENDTOTALLENGTH - (FILTER_L - 1)
#define PADDEDSINGLEINPUTLENGTH 2 * PADDEDNUMINPUTPOINTS
#define TOTALDATAINPUTLENGTH PADDEDSINGLEINPUTLENGTH *FILTER_NO;
#define TOTALFILTERSLENGTHKERNELARG FILTER_L *(FILTER_NO / TOTAL_WORK_ITEMS);
#define PADDEDNUMRESULTLENGTH 2 * (RESULT_L + EXTRAPADDING + PADDINGCOEFLOAD);

#define FILTER_LENGTH 32

#define FILTER_SIZE 128 * 64
#define INPUT_SIZE 541696
#define RESULT_SIZE 541696
#define INPUT_L 4096
#define FILTER_L 128
#define FILTER_NO 64

#define KNOB_COEF_SHIFT 32
#define KNOB_NUM_PARALLEL 8

#define KNOB_UNROLL_FILTER_1                                                   \
    1 // iterations: (FILTER_LENGTH-1)  -- data shift reg
#define KNOB_UNROLL_FILTER_2                                                   \
    1 // iterations: (NUM_COEF_LOADS-1) -- coef shift reg (outer)
#define KNOB_UNROLL_FILTER_3                                                   \
    1 // iterations: FILTER_LENGTH      -- computation (outer)
#define KNOB_UNROLL_COEF_SHIFT_1                                               \
    1 // iterations: KNOB_COEF_SHIFT    -- coef shift reg (inner)
#define KNOB_UNROLL_COEF_SHIFT_2                                               \
    1                            // iterations: KNOB_COEF_SHIFT    -- load coef
#define KNOB_UNROLL_TOTAL 1      // iterations: totalInputLength   -- main loop
#define KNOB_UNROLL_PARALLEL_1 1 // iterations: KNOB_NUM_PARALLEL  -- load data
#define KNOB_UNROLL_PARALLEL_2                                                 \
    1 // iterations: KNOB_NUM_PARALLEL  -- computation (inner)
#define KNOB_UNROLL_PARALLEL_3                                                 \
    1 // iterations: KNOB_NUM_PARALLEL  -- write result

#define KNOB_NUM_WORK_ITEMS 64
#define KNOB_NUM_WORK_GROUPS 1

#define KNOB_SIMD 1
#define KNOB_COMPUTE_UNITS 1

#define NUM_COEF_LOADS (FILTER_LENGTH / KNOB_COEF_SHIFT)
#define TOTAL_WORK_ITEMS (KNOB_NUM_WORK_ITEMS * KNOB_NUM_WORK_GROUPS)

#endif /* PARAMETERS_H_ */
