/* ----------------------------------------------------------------------
 * Name: tile_3D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp tile_3D(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
						float* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c)
{
	int w, h, c, i;

	for(c = 0; c < output_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				output[(w + h * output_w) * output_c + c] = input[((w % input_w) + (h % input_h) * input_w) * input_c + (c % input_c)];
			}
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}

tinyengine_status_fp tile_3D_IOHW(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
						float* output, const uint16_t mul_c)
{
	int w, h, c, i;

	for(i = 0; i < mul_c; i++){
		for(c = 0; c < input_c; c++){
			for(h = 0; h < input_h; h++){
				for(w = 0; w < input_w; w++){
					//output[((w + h * input_w) * input_c + c) + (i * input_c * input_w * input_h)] = input[(w + h * input_w) * input_c + c];
					output[((w + h * input_w) * input_c + c) + i * (input_h * input_w * input_c)] = input[(w + h * input_w) * input_c + c];
				}
			}
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}