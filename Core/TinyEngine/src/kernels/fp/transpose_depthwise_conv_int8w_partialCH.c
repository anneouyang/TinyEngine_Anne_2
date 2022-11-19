/* ----------------------------------------------------------------------
 * Name: transpose_depthwise_conv.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

tinyengine_status_fp TFLite_TransposeDepthwiseConv_int8w_partialCH(const Conv_Params params,
                 const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram,const int8_t* filter_flash,const uint16_t first_k_channel, const uint16_t filter_height, const uint16_t filter_width,
                 const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 float* im2col_data, const uint16_t batches) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Assume depth_multiplier = 1 //
  const int depth_multiplier = 1;

  /*
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  */
  (void)im2col_data;   // only used in optimized code.
  /*
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  */

  // Although transpose convolution simplifies to convolution with transposed
  // weights for strides of 1, non-unitary striding complicates matters. To
  // keep this reference implementation as clear as possible, we use a
  // "scatter" access pattern, where we loop through all the input elements,
  // computing their influence on the output, rather than looping through the
  // output elements in the typical "gather" access pattern of a conv. We
  // therefore must initialize the output array to zero.
  const int num_elements = output_height * output_width * output_depth;
  for (int i = 0; i < num_elements; i++) {
    output_data[i] = 0.0f;
  }

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          // Loop through the output elements it will influence
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int out_x_origin = (in_x * stride_width) - pad_width;
            const int out_y_origin = (in_y * stride_height) - pad_height;

            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) && (out_y < output_height)) {
                  float input_value = input_data[Offset(input_height, input_width, input_depth, batch, in_y, in_x, ic)];
                  int weight_offset;
                  int8_t filter_value_int8;
                  if (ic < first_k_channel){
                	  weight_offset = Offset(filter_height, filter_width, first_k_channel, 0, filter_y, filter_x, ic);
                	  filter_value_int8 = filter_sram[weight_offset];
                  }
				  else{
                	  weight_offset = Offset(filter_height, filter_width, input_depth - first_k_channel, 0, filter_y, filter_x, ic - first_k_channel);
                	  filter_value_int8 = filter_flash[weight_offset];
				  }

                  float filter_value = (float)filter_value_int8;
                  output_data[Offset(output_height, output_width, output_depth, batch, out_y, out_x, oc)] += input_value * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }

  if (bias_data) {
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            output_data[Offset(output_height, output_width, output_depth, batch, out_y, out_x, out_channel)] += bias_data[out_channel];
          }
        }
      }
    }
  }
}
