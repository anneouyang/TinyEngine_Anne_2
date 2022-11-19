/* ----------------------------------------------------------------------
 * Name: sum_2D.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp sum_2D(const float* input_data, const uint16_t matA_row,
                      const uint16_t matA_col, const uint16_t axis, float* output_data) {
  int i, j;
  float sum;

  if (axis == 0){
    for (i = 0; i < matA_row; ++i) {
      sum = 0;

      for (j = 0; j < matA_col; ++j) {
        sum += input_data[j + (i * matA_row)];
      }

      output_data[i] = sum;
    }
  }
  else{ /* axis == 1 */
    for (j = 0; j < matA_col; ++j) {
      sum = 0;

      for (i = 0; i < matA_row; ++i) {
        sum += input_data[j + (i * matA_row)];
      }

      output_data[j] = sum;
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
