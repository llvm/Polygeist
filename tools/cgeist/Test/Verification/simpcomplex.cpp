
struct mc {
    float r, i;
};

void foo1() {
    mc a;
}
void foo() {
    __complex__ float a;
}

mc *bar1() {
    auto a = new mc;
    return a;
}
__complex__ float *bar() {
    auto a = new __complex__ float;
    return a;
}

float access_imag1(mc a) {
    return a.i;
}
float access_real1(mc a) {
    return a.r;
}
float access_imag(__complex__ float a) {
    return __imag__ a;
}
float access_real(__complex__ float a) {
    return __real__ a;
}
float ref_imag(__complex__ float a) {
    __imag__ a = 2.0f;
    return __imag__ a;
}
float ref_real(__complex__ float a) {
    __real__ a = 3.0f;
    return __real__ a;
}
double cast(__complex__ float a) {
    __complex__ double b = a;
    return __real__ b + __imag__ b;
}
float imag_literal() {
    __complex__ float b = 10.0f + 3.0fi;
    return __imag__ b + __real__ b;
}
