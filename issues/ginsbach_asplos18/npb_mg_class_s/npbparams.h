/* CLASS = S */
/* Reproducible NPB MG Class-S configuration for cross-silicon validation. */
#define NX_DEFAULT     32
#define NY_DEFAULT     32
#define NZ_DEFAULT     32
#define NIT_DEFAULT    4
#define LM             5
#define LT_DEFAULT     5
#define DEBUG_DEFAULT  0
#define NDIM1          5
#define NDIM2          5
#define NDIM3          5
#define ONE            1

#define CONVERTDOUBLE  false
#define COMPILETIME "05 Sep 2026"
#define NPBVERSION "3.3.1"
#define CS1 "aarch64-linux-gnu-gcc"
#define CS2 "aarch64-linux-gnu-gcc"
#define CS3 "-lm"
#define CS4 "-I../common"
#define CS5 "-O3"
#define CS6 "-O3"
#define CS7 "randdp"
