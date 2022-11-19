/* ----------------------------------------------------------------------
 * Name: nll_loss.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp nll_loss(const float* input_data, const uint16_t input_dim, const uint16_t input_depth, 
                       const float* target, const uint16_t target_size, float* output_data) {
  int idx;

  for(int i = 0; i < target_size; i++){
	  if (target[i] > 0){
		  idx = i;
		  break;
	  }
  }

  output_data[0] = -input_data[idx];
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
