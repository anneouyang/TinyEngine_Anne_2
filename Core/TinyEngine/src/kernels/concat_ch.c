/*
 * concat_ch.c
 *
 *  Created on: May 15, 2021
 *      Author: wmchen
 */
#include "arm_nnfunctions.h"
#include "tinyengine_function.h"

tinyengine_status concat_ch(const q7_t *input1, const uint16_t input_x,
	const uint16_t input_y, const uint16_t input1_ch, const q7_t* input2, const uint16_t input2_ch, q7_t *output) {

	int elements = input_y * input_x;

	while(elements--){
		//place the first input
		memcpy(output, input1, input1_ch);
		input1 += input1_ch; output += input1_ch;

		//place the second input
		memcpy(output, input2, input2_ch);
		input2 += input2_ch; output += input2_ch;
	}

	return STATE_SUCCESS;
}





