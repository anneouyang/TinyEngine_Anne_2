/* ----------------------------------------------------------------------
 * Name: relu.c
 * Project: TinyEngine
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp relu(const uint16_t size, const float* input_data, float* output_data) {
  int i;

  for (i = 0; i < size; ++i) {
    output_data[i] = input_data[i] > 0 ? input_data[i] : 0;
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
