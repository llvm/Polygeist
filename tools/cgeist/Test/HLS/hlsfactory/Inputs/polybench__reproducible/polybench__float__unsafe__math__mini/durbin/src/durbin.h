#pragma once
#include <cmath>


extern "C" {
void kernel_durbin(
		   double r[ 40 + 0],
		   double y[ 40 + 0]);
}