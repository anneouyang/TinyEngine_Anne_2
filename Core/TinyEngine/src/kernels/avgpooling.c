/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */
#include "tinyengine_function.h"

tinyengine_status avg_pooling(const q7_t* input, const uint16_t input_h, const uint16_t input_w,
		const uint16_t input_c,	const uint16_t sample_h, const uint16_t sample_w,
		const uint16_t output_h, const uint16_t output_w, const int32_t out_activation_min,
        const int32_t out_activation_max, q7_t* output)
{
	int h, w, c;
	int sh, sw;
	const int divider_half = ((sample_h * sample_w) / 2);
	for(c = 0; c < input_c; c++){
		for(h = 0; h < output_h; h++){
			for(w = 0; w < output_w; w++){
				int avg = 0;

				for(sh = 0; sh < sample_h; sh++){
					int height = sh + h * sample_h;
					for(sw = 0; sw < sample_w; sw++){
						int width = sw + w * sample_w;
						avg += input[(width + height * input_w) * input_c + c];
					}
				}

				// for rounded div
				if (avg > 0)
					avg += divider_half;
				else
					avg -= divider_half;

				int out = avg / (sample_h * sample_w);
				out = TN_MAX(out, out_activation_min);
				out = TN_MIN(out, out_activation_max);
				output[(w + h * output_w) * input_c + c] = out;
			}
		}
	}
}


