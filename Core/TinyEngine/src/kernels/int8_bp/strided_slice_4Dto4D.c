/* ----------------------------------------------------------------------
 * Name: strided_slice_4Dto4D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"


//tinyengine_status strided_slice_4Dto4D(const q7_t* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c1, const uint16_t input_c2,
//                                          const uint16_t* begin, const uint16_t* end, const uint16_t* stride,
//                                          q7_t* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c1, const uint16_t output_c2)
//{
//  int h, w, c1, c2;

///* Assume this op is executed after "reshape" from 3D(HWC) to 4D(HWC1C2)
//   and "transpose" from HWC1C2 to HWC2C1*/
//  for(c2 = begin[3]; c2 < end[3]; c2 += stride[0]){
//    for(c1 = begin[2]; c1 < end[2]; c1 += stride[0]){
//      for(h = begin[1]; h < end[1]; h += stride[0]){
//        for(w = begin[0]; w < end[0]; w += stride[0]){
//          output[((w + h * output_w) * output_c1 + c1) * output_c2 + c2] = input[((w + h * input_w) * input_c1 + c1) * input_c2 + c2];
//        }
//      }
//    }
//  }
tinyengine_status strided_slice_4Dto4D_int8(const q7_t* input, const uint16_t inn, const uint16_t inc, const uint16_t inh, const uint16_t inw,
                                          const uint16_t* begin, const uint16_t* end, const uint16_t* stride,
                                          q7_t* output, const uint16_t on, const uint16_t oc, const uint16_t oh, const uint16_t ow)
{
  int n, c, h, w;
  //begin and end are in [n, c, h, w]
  for(n = begin[0]; n < end[0]; n += stride[0]){
		for(h = begin[2]; h < end[2]; h += stride[0]){
		  for(w = begin[3]; w < end[3]; w += stride[0]){
				for(c = begin[1]; c < end[1]; c += stride[0]){
					output[((h + n * oh) * ow + w) * oc + c] = input[((h + n * inh) * inw + w) * inc + c];
				}
			}
	  }
	}
	
	/* Return to application */
	return STATE_SUCCESS;
}
