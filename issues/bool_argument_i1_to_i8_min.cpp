extern void takes_bool(bool);

void bool_argument_i1_to_i8_min(void *ptr) {
  takes_bool(ptr != nullptr);
}
