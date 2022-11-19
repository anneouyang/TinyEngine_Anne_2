/* ----------------------------------------------------------------------
 * Name: transpose_3Dto3D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp transpose_3Dto3D(const float* input, const uint16_t input_h, const uint16_t input_w,
                                  const uint16_t input_c, float* output)
{
  int h, w, c;
  for(c = 0; c < input_c; c++){
    for(h = 0; h < input_h; h++){
      for(w = 0; w < input_w; w++){
        output[(h + w * input_h) * input_c + c] = input[(w + h * input_w) * input_c + c];
      }
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
