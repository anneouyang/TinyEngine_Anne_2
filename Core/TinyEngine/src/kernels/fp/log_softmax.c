/* ----------------------------------------------------------------------
 * Name: log_softmax.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

// Data is required to be contiguous, and so many operators can use either the
// full array flat size or the flat size with one dimension skipped (commonly
// the depth).
//template <int N>
int FlatSizeSkipDim(const uint16_t input_height, const uint16_t input_width) {
  //TFLITE_DCHECK(skip_dim >= 0 && skip_dim < N);
  int flat_size = input_height * input_width;
  /*
  for (int i = 0; i < 2; ++i) {
    flat_size *= (i == skip_dim) ? 1 : dims.sizes[i];
  }
  */
  return flat_size;
}

// A combination of MatchingFlatSize() and FlatSizeSkipDim().
//template <int N>
int MatchingFlatSizeSkipDim(const uint16_t input_height, const uint16_t input_width) {
  /*
  for (int i = 0; i < N; ++i) {
    if (i != skip_dim) {
      TFLITE_DCHECK_EQ(ArraySize(dims, i), ArraySize(check_dims_0, i));
    }
  }
  */
  return FlatSizeSkipDim(input_height, input_width);
}

int MatchingDim(const uint16_t data_A, const uint16_t data_B) {
  //TFLITE_DCHECK_EQ(shape1.Dims(index1), shape2.Dims(index2));
  return MIN(data_A, data_B);
}


tinyengine_status_fp LogSoftmax(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                       float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth) {
  //const int trailing_dim = input_dim - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_height, input_width);
  const int depth =
      MatchingDim(input_depth, output_depth);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    float max = FLT_MIN;
    for (int c = 0; c < depth; ++c) {
      max = MAX(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += exp(input_data[i * depth + c] - max);
    }

    // Compute result.
    const float log_sum = log(sum);
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = input_data[i * depth + c] - max - log_sum;
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
