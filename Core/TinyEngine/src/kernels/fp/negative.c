/* ----------------------------------------------------------------------
 * Name: negative.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp negative(const uint16_t size, const float* input1_data, bool* output_data) {
  int i;

  for (i = 0; i < size; ++i) {
    output_data[i] = input1_data[i] < 0;
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
