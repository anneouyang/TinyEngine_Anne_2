/* ----------------------------------------------------------------------
 * Name: depthwise_conv.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
/*
int Offset(const uint16_t dims_data1, const uint16_t dims_data2, const uint16_t dims_data3, int i0, int i1, int i2, int i3) {
  return ((i0 * dims_data1 + i1) * dims_data2 + i2) * dims_data3 + i3;
}

float ActivationFunctionWithMinMax(float x, float output_activation_min,
                                      float output_activation_max) {
  return MIN(MAX(x, output_activation_min), output_activation_max);
}
*/

tinyengine_status_fp TFLite_DepthwiseConv(const Depthwise_Params params,
                 const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
                 const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 float* im2col_data, const uint16_t batches) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  /*
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  */

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            float total = 0.f;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value =
                      input_data[Offset(input_height, input_width, input_depth, b, in_y, in_x, ic)];
                  float filter_value = filter_data[Offset(
                      filter_height, filter_width, input_depth, 0, filter_y, filter_x, oc)];
                  total += (input_value * filter_value);
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data) {
              bias_value = bias_data[oc];
            }
            output_data[Offset(output_height, output_width, output_depth, b, out_y, out_x, oc)] =
                ActivationFunctionWithMinMax(total + bias_value,
                                             output_activation_min,
                                             output_activation_max);
          }
        }
      }
    }
  }
}

tinyengine_status_fp TFLite_DepthwiseConv_int8_PerChannel(const Depthwise_Params params,
                 const int32_t* output_multiplier, const int32_t* output_shift,
                 const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
                 const int8_t* filter_data, const uint16_t filter_height, const uint16_t filter_width,
                 const int32_t* bias_data, 
                 int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
                 const uint16_t batches) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t total = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32_t input_value =
                      input_data[Offset(input_height, input_width, input_depth, b, in_y, in_x, ic)];
                  int32_t filter_value = filter_data[Offset(
                      filter_height, filter_width, input_depth, 0, filter_y, filter_x, oc)];

                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8_t, even though
                  // it is represented using int32_t. int32_t += int8_t *
                  // (int8_t - int8_t) so the highest value we can get from each
                  // accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(b/174275578): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  total += filter_value * (input_value + input_offset);
                }
              }
            }
            if (bias_data) {
              total += bias_data[oc];
            }

            total = MultiplyByQuantizedMultiplier(
                total, output_multiplier[oc], output_shift[oc]);
            total += output_offset;
            total = MAX(total, output_activation_min);
            total = MIN(total, output_activation_max);
            output_data[Offset3(output_height, output_width, output_depth, b, out_y, out_x, oc)]=
                (int8_t)(total);
          }
        }
      }
    }
  }
}

tinyengine_status_fp TFLite_DepthwiseConv_int8_PerChannel_partialCH(const Depthwise_Params params,
                 const int32_t* output_multiplier, const int32_t* output_shift,
                 const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
                 const int8_t* filter_sram,const int8_t* filter_flash,const uint16_t first_k_channel, const uint16_t filter_height, const uint16_t filter_width,
                 const int32_t* bias_data, 
                 int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
                 const uint16_t batches) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t total = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32_t input_value =
                      input_data[Offset(input_height, input_width, input_depth, b, in_y, in_x, ic)];

                  int weight_offset;
                  int8_t filter_value_int8;
                  if (oc < first_k_channel){
                    weight_offset = Offset(filter_height, filter_width, first_k_channel, 0, filter_y, filter_x, oc);
                    filter_value_int8 = filter_sram[weight_offset];
                  }
                  else{
                    weight_offset = Offset(filter_height, filter_width, input_depth - first_k_channel, 0, filter_y, filter_x, oc - first_k_channel);
                    filter_value_int8 = filter_flash[weight_offset];
                  }

                  int32_t filter_value = (int32_t)filter_value_int8;

                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8_t, even though
                  // it is represented using int32_t. int32_t += int8_t *
                  // (int8_t - int8_t) so the highest value we can get from each
                  // accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(b/174275578): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  total += filter_value * (input_value + input_offset);
                }
              }
            }
            if (bias_data) {
              total += bias_data[oc];
            }

            total = MultiplyByQuantizedMultiplier(
                total, output_multiplier[oc], output_shift[oc]);
            total += output_offset;
            total = MAX(total, output_activation_min);
            total = MIN(total, output_activation_max);
            output_data[Offset3(output_height, output_width, output_depth, b, out_y, out_x, oc)]=
                (int8_t)(total);
          }
        }
      }
    }
  }
}
