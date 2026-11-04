

#define PRAGMA_SUB(x) _Pragma(#x)
#define PRAGMA_HLS(x) PRAGMA_SUB(x)

#define rows 480
#define cols 640
#define KNOB_WINDOW_SIZE_X 16
#define inner_unroll1 1
#define inner_unroll2 1
#define outer_unroll 1
#define partition_factor 2
