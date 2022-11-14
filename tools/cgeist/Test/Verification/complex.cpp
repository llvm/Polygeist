// RUN: cgeist %s --function='*' -S | FileCheck %s

int foo() {
    __complex__ float a = 1.0i + 30.0f;
    a = a + a;
    float b = __real__ a;
    float c = __imag__ a;
    return (int) b + (int) c;
    // CHECK: foo
    // CHECK-NEXT: %[[c:.*]] = arith.constant 62
    // CHECK-NEXT: return %[[c]]
}

int bar() {
    __complex__ float a = 1.0i + 30.0f;
    a += a;
    float r = __real__ a + __imag__ a;
    return r;
    // CHECK: bar
    // CHECK-NEXT: %[[c:.*]] = arith.constant 62
    // CHECK-NEXT: return %[[c]]
}

class mcomplex
{
    public:
        typedef double value_type;
        typedef __complex__ double _ComplexT;


        mcomplex(double __r = 0.0, double __i = 0.0)
        : _M_value{ __r, __i } { }
        //{
        //    __real__ _M_value = __r;
        //    __imag__ _M_value = __i;
        //}

    private:
        _ComplexT _M_value;

};

//mcomplex baz() {
//    mcomplex a(1, 30);
//    return a;
//}
