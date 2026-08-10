struct recursive_function_pointer_record_min;

typedef void (*recursive_function_pointer_record_min_callback)(
    struct recursive_function_pointer_record_min *);

struct recursive_function_pointer_record_min {
  recursive_function_pointer_record_min_callback callback;
};

void recursive_function_pointer_record_min_call(
    struct recursive_function_pointer_record_min *value) {
  if (value->callback) {
    value->callback(value);
  }
}
