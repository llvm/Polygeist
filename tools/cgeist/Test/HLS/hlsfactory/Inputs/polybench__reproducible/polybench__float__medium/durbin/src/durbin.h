#pragma once
#include <cmath>


extern "C" {
void kernel_durbin(
		   double r[ 400 + 0],
		   double y[ 400 + 0]);
}