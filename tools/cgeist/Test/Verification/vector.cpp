// RUN: cgeist %s --function='*' -S | FileCheck %s
// TODO:
// XFAIL: *

typedef int v4si __attribute__ ((vector_size (16)));

int foo() {
    v4si a = {1,2,3,4};
    v4si b = {10,20,30,40};
    a = b + 1;    /* a = b + {1,1,1,1}; */
    a = 2 * b;    /* a = {2,2,2,2} * b; */

    return a[1];
}

int bar() {
    v4si a = {1,2,3,4};
    v4si b = {10,20,30,40};
    v4si c;

    c = a >  b;     /* The result would be {0, 0,-1, 0}  */
    c = a == b;     /* The result would be {0,-1, 0,-1}  */
    return c[3];
}
