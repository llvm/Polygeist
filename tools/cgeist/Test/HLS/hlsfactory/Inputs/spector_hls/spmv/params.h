#define num_rows 512
#define UNROLL_F 4
#define outer_unroll 1
#define inner_unroll1 2
#define inner_unroll2 2
#define array_partition1 2
#define array_partition2 2
#define PRAGMA_SUB(x) _Pragma(#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

#if UNROLL_F == 1
const int loop_tripcount1 = 3;
const int loop_tripcount2 = 0;
#endif

#if UNROLL_F == 2
const int loop_tripcount1 = 2;
const int loop_tripcount2 = 1;
#endif

#if UNROLL_F > 2
const int loop_tripcount1 = 1;
const int loop_tripcount2 = 3;
#endif
