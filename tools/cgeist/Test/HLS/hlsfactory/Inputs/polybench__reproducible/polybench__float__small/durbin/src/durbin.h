#pragma once
#include <cmath>


extern "C" {
void kernel_durbin(
		   double r[ 120 + 0],
		   double y[ 120 + 0]);
}