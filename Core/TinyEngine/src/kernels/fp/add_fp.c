/* ----------------------------------------------------------------------
 * Name: add.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp add_fp(const uint16_t size, const float* input1_data,
			               const float* input2_data, float* output_data) {
  int i;

  for (i = 0; i < size; ++i) {
    output_data[i] = input1_data[i] + input2_data[i];
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
