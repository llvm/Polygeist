

struct wrap {
    int a, b, c;
};

//struct wrap {
//    threeInt t;
//};

wrap *foo() {
    auto a = new wrap;
    return a;
}

int bar() {
    wrap a;
    wrap b;
    a.a = 1;
    b.a = 2;

    return a.a + b.a;
}
