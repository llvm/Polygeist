#include "aes.h"

int main() {
  aes256_context ctx_obj;
  aes256_context *ctx = &ctx_obj;
  unsigned char k[32];
  unsigned char buf[16];
  aes256_encrypt_ecb(ctx, k, buf);
  return 0;
}
