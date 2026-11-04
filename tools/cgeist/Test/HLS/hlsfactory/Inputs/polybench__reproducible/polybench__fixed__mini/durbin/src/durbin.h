#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_durbin(
		   t_ap_fixed r[ 40 + 0],
		   t_ap_fixed y[ 40 + 0]);
}