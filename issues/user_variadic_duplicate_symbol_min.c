void user_variadic_duplicate_symbol_min_sink(const char *fmt, ...) {
}

void user_variadic_duplicate_symbol_min(int x) {
  user_variadic_duplicate_symbol_min_sink("%d", x);
}
