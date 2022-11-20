// RUN: cgeist %s --struct-abi=0 --function='*' -S | FileCheck %s --check-prefix=STRUCT
// COM: we dont support this yet RUN: cgeist %s --function='*' -S | FileCheck %s

struct mc {
    float r, i;
};

void hmm(mc a) {
    a.r = 10;
}
float um() {
    mc a;
    a.r = 50;
    hmm(a);
    return a.r;
}
int main() {
    return (int)um();
}

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
float imag_literal2() {
    __complex__ float b = 3.0fi;
    return __imag__ b + __real__ b;
}
float add() {
    __complex__ float a = 10.0f + 5.0fi;
    __complex__ float b = 30.0f + 2.0fi;
    __complex__ float c = a + b;
    return __imag__ c + __real__ c;
}
float addassign() {
    __complex__ float a = 10.0f + 5.0fi;
    __complex__ float c = 30.0f + 2.0fi;
    c += a;
    return __imag__ c + __real__ c;
}
class mcomplex
{
    public:
        mcomplex(double __r, double __i)
        : _M_value{ __r, __i } {}
    private:
        __complex__ double _M_value;
};
mcomplex *baz() {
    mcomplex *a = new mcomplex(1, 30);
    return a;
}
