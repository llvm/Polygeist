struct Incomplete;

struct Holder {
  Incomplete *ptr;
};

void use_holder(Holder h) { (void)h.ptr; }

