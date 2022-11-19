/*
 * avgpooling.c
 *
 *  Created on: Apr 17, 2021
 *      Author: wmchen
 */

#include "tinyengine_function.h"

tinyengine_status max_pooling(const q7_t* input, const uint16_t input_h, const uint16_t input_w,
		const uint16_t input_c,	const uint16_t sample_h, const uint16_t sample_w,
		const uint16_t output_h, const uint16_t output_w, const int32_t out_activation_min,
        const int32_t out_activation_max, q7_t* output)
{
	int h, w, c;
	int sh, sw;
	for(c = 0; c < input_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				int max = out_activation_min;

				for(sh = 0; sh < sample_h; sh++){
					int height = sh + h * sample_h;
					for(sw = 0; sw < sample_w; sw++){
						int width = sw + w * sample_w;
						max = TN_MAX(max,input[(width + height * input_w) * input_c + c]);
					}
				}

				int out = max;
				out = TN_MAX(out, out_activation_min);
				out = TN_MIN(out, out_activation_max);
				output[(w + h * output_w) * input_c + c] = out;
			}
		}
	}
}


