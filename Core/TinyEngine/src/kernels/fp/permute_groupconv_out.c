/* ----------------------------------------------------------------------
 * Name: permute_groupconv_out.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */
#include "tinyengine_function_fp.h"

tinyengine_status_fp permute_groupconv_out(float* input_data, const uint16_t d1, const uint16_t d2, const uint16_t d3, const uint16_t input_c, const uint16_t output_c,
                       float* output_data) {
  int h, w, c, cc;

  // 1, d1, d2, d3 (output_c*input_c) -> output_c, d1, d2, input_c
  for (cc = 0; cc < output_c; cc++){
    for(h = 0; h < d1; h++){
      for (w = 0; w < d2; w++){
        for (c = 0 ; c < input_c; c++){
          output_data[((((cc * d1 + h) * d2) + w) * input_c) + c] = input_data[((((h * d2 + w) * output_c) + cc) * input_c) + c];
        }
      }
    }
  }

  // inplace update
  int i, size = d1 * d2 * d3;
  for (i = 0; i < size; i++){
    input_data[i] = output_data[i];
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}