/* ----------------------------------------------------------------------
 * Name: tile_1D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp tile_1D(const float* input, const uint16_t input_size,
						const uint16_t* reps, const uint16_t reps_size, 
						float* output, const uint16_t* output_size)
{
	int w, h, c, i;

	/*if (reps_size > 1){
		T reshaped_input[reps_size];
		int reshaped_reps[reps_size];

		for (i=0; i<reps_size; i++){
			reshaped_reps[i] = reps[i];
			reshaped_input[i] = ;
		}
	}
	else{
		T reshaped_input[input_size];
		int reshaped_reps[reps_size];

		for (i=0; i<input_size; i++){
			reshaped_input[i] = input[i];
		}
		for (i=0; i<reps_size; i++){
			reshaped_reps[i] = reps[i];
		}
	}*/

	int output_length = sizeof(output_size) / sizeof(output_size[0]);

	if (output_length == 1){
		for (w=0; w<output_size[0]; w++){
			output[w] = input[w % input_size];
		}
	}
	else if (output_length == 2){
		for(h = 0; h < output_size[1]; h++){
			for(w = 0; w < output_size[0]; w++){
				output[w + h * output_size[0]] = input[w % input_size];
				//output[w + h * output_size[0]] = input[(w + h * output_size[0]) % input_size];
			}
		}
	}
	else{ /* output_length == 3 */
		for(c = 0; c < output_size[2]; c++){
			for(h = 0; h < output_size[1]; h++){
				for(w = 0; w < output_size[0]; w++){
					output[(w + h * output_size[0]) * output_size[3] + c] = input[w % input_size];
					//output[(w + h * output_size[0]) * output_size[3] + c] = input[((w + h * output_size[0]) * output_size[3] + c) % input_size];
				}
			}
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}
