/* ----------------------------------------------------------------------
 * Name: tile_2D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp tile_2D(const float* input, const uint16_t input_x, const uint16_t input_y,
						const uint16_t* reps, const uint16_t reps_size,
						float* output, const uint16_t* output_size)
{
	int w, h, c, i;
	int output_length = sizeof(output_size) / sizeof(output_size[0]);

	if (output_length == 2){
		for(h = 0; h < output_size[1]; h++){
			for(w = 0; w < output_size[0]; w++){
				output[w + h * output_size[0]] = input[(w % input_x) + (h % input_y) * input_x];
			}
		}
	}
	else{ /* output_length == 3 */
		for(c = 0; c < output_size[2]; c++){
			for(h = 0; h < output_size[1]; h++){
				for(w = 0; w < output_size[0]; w++){
					output[(w + h * output_size[0]) * output_size[3] + c] = input[(w % input_x) + (h % input_y) * input_x];
				}
			}
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}
