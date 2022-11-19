/* ----------------------------------------------------------------------
 * Name: permute4D_dim3012.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */
#include "tinyengine_function_fp.h"

tinyengine_status_fp permute4D_dim3012(float* input_data, const uint16_t d1, const uint16_t d2, const uint16_t d3, const uint16_t d4,
                       float* output_data) {
  int h, w, c, cc;

  // 3012
  for (cc = 0; cc < d4; cc++){
    for(c = 0; c < d1; c++){
      for (h = 0; h < d2; h++){
        for (w = 0 ; w < d3; w++){
          output_data[((((cc * d1 + c) * d2) + h) * d3) + w] = input_data[((((c * d2 + h) * d3) + w) * d4) + cc];
        }
      }
    }
  }

  // inplace update
//  int i, size = d1 * d2 * d3 * d4;
//  for (i = 0; i < size; i++)
//    input_data[i] = output_data[i];
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}