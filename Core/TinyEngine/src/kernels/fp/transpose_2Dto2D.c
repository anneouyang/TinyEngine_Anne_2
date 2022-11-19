/* ----------------------------------------------------------------------
 * Name: transpose_2Dto2D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp transpose_2Dto2D(const float* input, const uint16_t input_h, const uint16_t input_w, float* output)
{
  int h, w;

  for(w = 0; w < input_w; w++){
    for(h = 0; h < input_h; h++){
        output[w * input_h + h] = input[h * input_w + w];
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
