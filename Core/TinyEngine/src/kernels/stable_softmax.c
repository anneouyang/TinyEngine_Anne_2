#include "tinyengine_function.h"
#include <float.h>
#include <math.h>

tinyengine_status statble_softmax_inplace(float *input, const uint16_t length)
{
	float max = FLT_MIN;
	float exp_sum = 0;
	uint16_t i;
	for (i = 0; i < length; i++){
		if (input[i] > max) max = input[i];
	}

	// inplace update
	for (i = 0; i < length; i++){
		input[i] = exp(input[i] - max);
		exp_sum += input[i];
	}
	for (i = 0; i < length; i++){
		input[i] = input[i] / exp_sum;
	}
}
