/* ----------------------------------------------------------------------
 * Name: permute3D_120.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */
#include "tinyengine_function_fp.h"

// HWC -> CHW (since the C is actually output channels instead of input channel for the following conv)
tinyengine_status_fp permute3D_dim120(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
                       float* output_data) {
	int h, w, c;

	for(c = 0; c < input_depth; c++){
		for (h = 0; h < input_height; h++){
			for (w = 0 ; w < input_width; w++){
				output_data[((c * input_height + h) * input_width) + w] = input_data[((h * input_width + w) * input_depth) + c];
			}
		}
	}

	// inplace update
	int i, size = input_height * input_width * input_depth;
	for (i = 0; i < size; i++){
		input_data[i] = output_data[i];
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}
