/* ----------------------------------------------------------------------
 * Name: where.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"

tinyengine_status where_int8(const bool* inMask, const uint16_t size, const q7_t* input1_data,
                     const q7_t* input2_data, q7_t* output_data) {
  int i;

  for (i = 0; i < size; ++i) {
    output_data[i] = inMask[i] > 0 ? input1_data[i] : input2_data[i];
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status where_zeros_int8(const bool* inMask, const uint16_t size, const q7_t* input1_data, q7_t* output_data) {
  int i;

  for (i = 0; i < size; ++i) {
    output_data[i] = inMask[i] > 0 ? input1_data[i] : 0;
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status where_zeros_int8_inplace(const bool* inMask, const uint16_t size, q7_t* input1_data) {
  int i;

  for (i = 0; i < size; ++i) {
	  input1_data[i] = inMask[i] > 0 ? input1_data[i] : 0;
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status where_zeros_int8_inplace_bit(const unsigned char* inMask, const uint16_t size, q7_t* input1_data) {
  int i;

  for (i = 0; i < size; ++i) {
	  int bit_starting_idx = i % 8;
	  int mask = BIT_CHECK(inMask[i/8], bit_starting_idx);
	  input1_data[i] = mask > 0 ? input1_data[i] : 0;
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}
