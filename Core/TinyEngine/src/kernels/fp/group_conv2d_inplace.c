/* ----------------------------------------------------------------------
 * Name: conv2d.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define CHN_PER_LOOP (8)

// TODO: test this function carefully
// Note: Currently, this is specialized for group conv used in bp, where in_ch == group
tinyengine_status_fp group_conv_fp_inplace(const Conv_Params params, int group,
                 const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t _input_depth,
                 const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
                 const float* bias_data, 
                 int8_t* weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t _output_depth,
                 float* im2col_data, const uint16_t batches,
                 const float* scales, const float learning_rate) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  /*
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  */

  /*
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  */

  /*
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  */

  float input_value, filter_value;

  int input_gdepth = _input_depth / group;
  int output_gdepth = _output_depth / group;
  for (int batch = 0; batch < batches; ++batch) {
	  for (int g = 0; g < group; ++g){
		for (int out_y = 0; out_y < output_height; ++out_y) {
		  const int in_y_origin = (out_y * stride_height) - pad_height;
		  for (int out_x = 0; out_x < output_width; ++out_x) {
			const int in_x_origin = (out_x * stride_width) - pad_width;
			for (int out_channel = 0; out_channel < output_gdepth; ++out_channel) {
			  float total = 0.f;
			  for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
				const int in_y = in_y_origin + dilation_height_factor * filter_y;
				for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
				  const int in_x = in_x_origin + dilation_width_factor * filter_x;

				  // Zero padding by omitting the areas outside the image.
				  const bool is_point_inside_image =
					  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
					  (in_y < input_height);

				  if (!is_point_inside_image) {
					continue;
				  }

				  int in_channel = 0;
				  for (; in_channel < input_gdepth; ++in_channel) {
					input_value = input_data[Offset3(input_height, input_width, _input_depth, batch, in_y, in_x, in_channel + input_gdepth * g)];
					// gorup x g_out_ch x kh * kw * g_inc
//					filter_value = filter_data[gOffset(output_gdepth, filter_height, filter_width, input_gdepth, 0, out_channel, filter_y, filter_x, in_channel)];
					// access it as normal conv since weights are repeatedly in group dimension in bp
					filter_value = filter_data[Offset3(filter_height, filter_width, output_gdepth, 0, filter_y, filter_x, out_channel)];
					total += (input_value * filter_value);
				  }
				}
			  }
			  float bias_value = 0.0f;
			  if (bias_data) {
				bias_value = bias_data[out_channel];
			  }
			  float gradient = _ActivationFunctionWithMinMax(total + bias_value,
					   output_activation_min,
					   output_activation_max);
			  int weigth_idx = Offset3(output_height, output_width, _output_depth, batch, out_y, out_x, out_channel + g * output_gdepth);
			  weight_data[weigth_idx] -= gradient * scales[out_channel] * learning_rate; //TODO: multipliy with scaler and learning rate
			}
		  }
		}
	  }
  }
}

tinyengine_status_fp group_conv_fp_int8input_inplace(const Conv_Params params, int group,
                 const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t _input_depth,
                 const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
                 const float* bias_data, 
                 int8_t* weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t _output_depth,
                 float* im2col_data, const uint16_t batches,
                 const float* scales, const float learning_rate) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  /*
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  */

  /*
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  */

  /*
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  */

  float input_value, filter_value;

  int input_gdepth = _input_depth / group;
  int output_gdepth = _output_depth / group;
  for (int batch = 0; batch < batches; ++batch) {
	  for (int g = 0; g < group; ++g){
		for (int out_y = 0; out_y < output_height; ++out_y) {
		  const int in_y_origin = (out_y * stride_height) - pad_height;
		  for (int out_x = 0; out_x < output_width; ++out_x) {
			const int in_x_origin = (out_x * stride_width) - pad_width;
			for (int out_channel = 0; out_channel < output_gdepth; ++out_channel) {
			  float total = 0.f;
			  for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
				const int in_y = in_y_origin + dilation_height_factor * filter_y;
				for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
				  const int in_x = in_x_origin + dilation_width_factor * filter_x;

				  // Zero padding by omitting the areas outside the image.
				  const bool is_point_inside_image =
					  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
					  (in_y < input_height);

				  if (!is_point_inside_image) {
					continue;
				  }

				  int in_channel = 0;
				  for (; in_channel < input_gdepth; ++in_channel) {
					input_value = (float)input_data[Offset3(input_height, input_width, _input_depth, batch, in_y, in_x, in_channel + input_gdepth * g)];
					// gorup x g_out_ch x kh * kw * g_inc
//					filter_value = filter_data[gOffset(output_gdepth, filter_height, filter_width, input_gdepth, 0, out_channel, filter_y, filter_x, in_channel)];
					// access it as normal conv since weights are repeatedly in group dimension in bp
					filter_value = filter_data[Offset3(filter_height, filter_width, output_gdepth, 0, filter_y, filter_x, out_channel)];
					total += (input_value * filter_value);
				  }
				}
			  }
			  float bias_value = 0.0f;
			  if (bias_data) {
				bias_value = bias_data[out_channel];
			  }
			  float gradient = _ActivationFunctionWithMinMax(total + bias_value,
					   output_activation_min,
					   output_activation_max);
			  int weigth_idx = Offset3(output_height, output_width, _output_depth, batch, out_y, out_x, out_channel + g * output_gdepth);
			  weight_data[weigth_idx] -= gradient * scales[out_channel] * learning_rate; //TODO: multipliy with scaler and learning rate
			}
		  }
		}
	  }
  }
}
