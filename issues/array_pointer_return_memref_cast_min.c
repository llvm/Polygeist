typedef unsigned char array_pointer_return_memref_cast_guid[16];
typedef array_pointer_return_memref_cast_guid
    *array_pointer_return_memref_cast_guid_t;

array_pointer_return_memref_cast_guid_t
array_pointer_return_memref_cast_min(void) {
  static array_pointer_return_memref_cast_guid guid = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return &guid;
}
