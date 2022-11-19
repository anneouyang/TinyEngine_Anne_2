/* ----------------------------------------------------------------------
 * Name: tte_exp.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp tte_exp(const uint16_t size, const float* input_data, float* output_data) {
  int i;
  
  for (i = 0; i < size; ++i) {
    output_data[i] = exp(input_data[i]);
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
