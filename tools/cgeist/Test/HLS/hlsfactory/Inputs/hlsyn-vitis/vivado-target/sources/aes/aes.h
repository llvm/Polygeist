#ifndef TOP_H
#define TOP_H


#include <string.h>
#include "ap_fixed.h"
#include <complex>
#include <hls_stream.h>
#include <iomanip>
#include <iostream>
#include <cmath>
#include "hls_vector.h"

typedef struct {unsigned char key[32];unsigned char enckey[32];unsigned char deckey[32];}aes256_context;
void aes256_encrypt_ecb(aes256_context *ctx,unsigned char k[32],unsigned char buf[16]);

#endif
