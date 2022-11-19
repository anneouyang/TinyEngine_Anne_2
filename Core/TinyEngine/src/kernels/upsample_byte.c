/*
 * upsample_byte.c
 *
 *  Created on: May 15, 2021
 *      Author: wmchen
 */
#include "arm_nnfunctions.h"
#include "tinyengine_function.h"

tinyengine_status upsample_byte(const q7_t *input, const uint16_t input_x,
	const uint16_t input_y, const uint16_t input_ch, q7_t *output, const uint16_t sample_factor) {
	//get output resolution
	const uint16_t output_x = input_x * sample_factor, output_y = input_y * sample_factor , output_ch = input_ch;

	//upsample in a repeated manner
	for(int ih = 0; ih < input_y; ih++){
		q7_t* out_head = output;
		//place 1 row
		for(int iw = 0; iw < input_x; iw++){
			for(int s = 0; s < sample_factor; s++){
				memcpy(output, input, input_ch);
				output += input_ch;
			}
			input += input_ch;
		}

		//copy the remaining rows
		for(int s = 1; s < sample_factor; s++){
			memcpy(output, out_head, output_ch * output_x);
			output += output_ch * output_x;
		}
	}
	return STATE_SUCCESS;
}


//ref: https://www.cs.toronto.edu/~guerzhoy/320/lec/upsampling.pdf
tinyengine_status upsample_byte_bilinear(const q7_t *input, const uint16_t input_x,
	const uint16_t input_y, const uint16_t input_ch, q7_t *output, const uint16_t sample_factor) {
	//get output resolution
	const uint16_t output_x = input_x * sample_factor, output_y = input_y * sample_factor , output_ch = input_ch;

//	//upsample in a repeated manner
//	for(int oh = 0; oh < input_y; oh++){
//		int ih = oh / sample_factor;
//		int rh = oh % sample_factor;
//
//		q7_t* out_head = output;
//		//place 1 row
//		for(int ow = 0; ow < onput_x; ow++){
//			int iw = iw / sample_factor;
//			int wh = wh % sample_factor;
//
//			//exact coordinate
//			q7_t* ori_input = input + input_ch * (input_x * ih + iw);
//			if(rh | wh == 0){
//				memcpy(output, ori_input, input_ch);
//				continue;
//			}
//
//			//interpolate
//			q7_t* topleft = ori_input;
//			q7_t* topright = ori_input + input_ch;
//			q7_t* bottomleft = topleft + input_ch * input_x;
//			q7_t* bottomright = topright + input_ch * input_x;
//		}
//	}
	return STATE_SUCCESS;
}




