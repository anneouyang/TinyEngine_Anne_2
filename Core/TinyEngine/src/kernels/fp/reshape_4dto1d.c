/* ----------------------------------------------------------------------
 * Name: reshape_4dto1d.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp reshape_3dto1d(const float* input, const uint16_t h, const uint16_t w, const uint16_t c, float* output)
{
  uint16_t output_c = h * w * c;
  int i,j,k;

  //HWC -> CHW alignment
  for (i = 0; i < c; i++){
    for (j = 0; j < h; j++){
      for (k= 0; k < w; k++){
        *output++ = input[(k + w * j)* c + i];
      }
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}