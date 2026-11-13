using callback_t = void (*)(void *, void *, int, char *);

void callback_impl(void *, void *, int, char *) {}

callback_t choose_callback(bool use_null) {
  return use_null ? nullptr : callback_impl;
}

