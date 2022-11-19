/* ----------------------------------------------------------------------
 * Name: sum_4D_exclude.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"

tinyengine_status sum_4D_exclude_int8(const q7_t* input_data, const uint16_t d1, const uint16_t d2,
                      const uint16_t d3, const uint16_t d4, const uint16_t axis, q31_t* output_data) {
  int i, j, m, n;

  if (axis == 0){
    for (i = 0; i < d1; i++){
      q31_t sum = 0;
      for (j = 0; j < d2; j++){
        for (m = 0; m < d3; m++) {
          for (n = 0; n < d4; n++){
            sum += input_data[((i * d2 + j) * d3 + m) * d4 + n];
          }
        }
      }
      output_data[i] = sum;
    }
  }
  else if (axis == 1){
    for (j = 0; j < d2; j++){
      q31_t sum = 0;
      for (i = 0; i < d1; i++){
        for (m = 0; m < d3; m++) {
          for (n = 0; n < d4; n++){
            sum += input_data[((i * d2 + j) * d3 + m) * d4 + n];
          }
        }
      }
      output_data[j] = sum;
    }
  }
  else if (axis == 2){
    for (m = 0; m < d3; m++) {
      q31_t sum = 0;
        for (i = 0; i < d1; i++){
          for (j = 0; j < d2; j++){
          for (n = 0; n < d4; n++){
            sum += input_data[((i * d2 + j) * d3 + m) * d4 + n];
          }
        }
      }
      output_data[m] = sum;
    }
  }
  else if (axis == 3){
    for (n = 0; n < d4; n++){
      q31_t sum = 0;
        for (i = 0; i < d1; i++){
          for (j = 0; j < d2; j++){
            for (m = 0; m < d3; m++) {
            sum += input_data[((i * d2 + j) * d3 + m) * d4 + n];
          }
        }
      }
      output_data[n] = sum;
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}