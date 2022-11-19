/* ----------------------------------------------------------------------
 * Name: reshape_like_1dto4d.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp reshape_like_1dto4d(const float* input, const uint16_t h, const uint16_t w, const uint16_t c, float* output)
{
  uint16_t output_c = h * w * c;
  int i,j,k;

  //CHW -> HWC alignment
  for (i = 0; i < h; i++){
    for (j = 0; j < w; j++){
      for (k= 0; k < c; k++){
        *output++ = input[(i + h * k)* w + j];
      }
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}