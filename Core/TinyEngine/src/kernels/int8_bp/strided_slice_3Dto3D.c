/* ----------------------------------------------------------------------
 * Name: strided_slice_3Dto3D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"

tinyengine_status strided_slice_3Dto3D_int8(const q7_t* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
                                          const uint16_t* begin, const uint16_t* end, const uint16_t* stride, 
                                          q7_t* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c)
{
  int h, w, c;
  
  for(c = begin[2]; c < end[2]; c += stride[2]){
    for(h = begin[1]; h < end[1]; h += stride[1]){
      for(w = begin[0]; w < end[0]; w += stride[0]){
        output[(w + h * output_w) * output_c + c] = input[(w + h * input_w) * input_c + c];
      }
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}
