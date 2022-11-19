/* ----------------------------------------------------------------------
 * Name: bias_add_2D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp bias_add_2D(const float* input, const uint16_t input_h, const uint16_t input_w, 
		const float* bias, float* output)
{
	int h, w;
	for(h = 0; h < input_h; h++){
		for(w = 0; w < input_w; w++){
			output[h * input_w + w] = input[h * input_w + w] + bias[w];
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}
