/* ----------------------------------------------------------------------
 * Project: TinyEngine, MCUNetV3
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.ed
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Wei-Chen Wang, wweichen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#include <stdint.h>
//#include "genInclude.h"
//#include "fp_requantize_op.h"
#include <complex.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

typedef int8_t q7_t;
typedef uint8_t q8_t;
typedef int16_t q15_t;
typedef uint16_t q16_t;
typedef int32_t q31_t;
typedef uint32_t q32_t;

typedef enum {
	STATE_SUCCESS_fp = 0, /* No error */
	PARAM_NO_SUPPORT_fp = 1, /* Unsupported parameters */
} tinyengine_status_fp;

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

//enum class PaddingType : uint8_t { kNone, kSame, kValid };
typedef struct padding_values {
  int16_t width;
  int16_t height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16_t width_offset;
  // Same as width_offset except it's over the height dimension.
  int16_t height_offset;
} Padding_Values;

typedef struct conv_params {
  //PaddingType padding_type;
  Padding_Values padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
} Conv_Params;

typedef struct depthwise_params {
  //PaddingType padding_type;
  Padding_Values padding_values;
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  int16_t depth_multiplier;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  const int32_t* output_multiplier_per_channel;
  const int32_t* output_shift_per_channel;
} Depthwise_Params;

typedef struct transpose_params {
  int8_t perm_count;
  int32_t perm[5];
} Transpose_Params;


tinyengine_status_fp relu(const uint16_t size, const float* input_data, float* output_data);

tinyengine_status_fp bias_add_2D(const float* input, const uint16_t input_x, const uint16_t input_y,
		const float* bias, float* output);

tinyengine_status_fp bias_add_3D(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
		const float* bias, float* output);

tinyengine_status_fp mat_mul(const float* matA, const uint16_t matA_row, const uint16_t matA_col,
		const float* matB, const uint16_t matB_col, float* output);

tinyengine_status_fp dense(const float* matA, const uint16_t matA_row, const uint16_t matA_col,
		const float* matB, const uint16_t matB_row, float* output);

tinyengine_status_fp tile_1D(const float* input, const uint16_t input_size,
		const uint16_t* reps, const uint16_t reps_size, 
		float* output, const uint16_t* output_size);

tinyengine_status_fp tile_2D(const float* input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t* reps, const uint16_t reps_size,
		float* output, const uint16_t* output_size);

tinyengine_status_fp tile_3D(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
						float* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c);

tinyengine_status_fp tile_3D_IOHW(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
						float* output, const uint16_t mul_c);

tinyengine_status_fp less(const uint16_t size, const float* input1_data,
		const float* input2_data, bool* output_data);

tinyengine_status_fp where(const bool* inMask, const uint16_t size, const float* input1_data,
		const float* input2_data, float* output_data);

tinyengine_status_fp where_zeros(const bool* inMask, const uint16_t size, const float* input1_data, float* output_data);

tinyengine_status_fp where_zeros_inplace(const bool* inMask, const uint16_t size, float* input1_data);

tinyengine_status_fp where_zeros_inplace_bit(const unsigned char* inMask, const uint16_t size, float* input1_data);

tinyengine_status_fp negative(const uint16_t size, const float* input1_data,
		bool* output_data);

tinyengine_status_fp sum_2D(const float* input_data, const uint16_t matA_row,
		const uint16_t matA_col, const uint16_t axis, float* output_data);

tinyengine_status_fp sum_3D(const float* input_data, const uint16_t input_w, const uint16_t input_h,
		const uint16_t input_c, const uint16_t axis, float* output_data);

tinyengine_status_fp add_fp(const uint16_t size, const float* input1_data,
		const float* input2_data, float* output_data);

tinyengine_status_fp sub(const uint16_t size, const float* input1_data,
		const float* input2_data, float* output_data);

tinyengine_status_fp mul(const uint16_t size, const float* input1_data,
		const float* input2_data, float* output_data);

tinyengine_status_fp div_fp(const uint16_t size, const float* input1_data,
		const float* input2_data, float* output_data);

tinyengine_status_fp tte_exp(const uint16_t size, const float* input_data,
		float* output_data);

tinyengine_status_fp TFLite_Conv_fp(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const uint16_t filter_input_depth,
		const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_Conv_int8_PerChannel(
    const Conv_Params params, const int32_t* output_multiplier,
    const int32_t* output_shift, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
    const int8_t* input_data, 
    const int8_t* filter_data, const uint16_t filter_height, const uint16_t filter_width, const uint16_t filter_input_depth,
    const int32_t* bias_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
    int8_t* output_data, const uint16_t batches);

tinyengine_status_fp TFLite_Conv_int8_PerChannel_partialCH(
    const Conv_Params params, const int32_t* output_multiplier,
    const int32_t* output_shift, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
    const int8_t* input_data, 
    const int8_t* filter_sram,const int8_t* filter_flash,const uint16_t first_k_channel, 
    const uint16_t filter_height, const uint16_t filter_width, const uint16_t filter_input_depth,
    const int32_t* bias_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
    int8_t* output_data, const uint16_t batches);

tinyengine_status_fp conv_fp_kernel3_stride2_pad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp conv_fp_kernel3_stride1_pad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp conv_fp_kernel3_stride1_pad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp conv_fp_kernelx_stride1_pad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp conv_fp_kernelx_stride1_pad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp group_conv_fp_kernelx_stride1_pad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_int8input_inplace_revised(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_int8input_inplace_revised(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_2row32col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace_revised(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_inplace_revised(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_2row32col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_int8input_inplace_revised(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_int8input_inplace_revised(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_2row32col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_inplace_revised(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_inplace_revised(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel4_stride1_pad0_in4x4_out1x1_uniweight_2row32col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row8col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row16col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_2row32col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row8col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row16col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_2row32col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row8col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row8col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row16col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row16col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row16col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_4row4col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_2row32col_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_2row32col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_2row32col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_2row16col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_1row32col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_1row16col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_1row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_conv_fp_kernel3_stride1_pad0_in3x3_out1x1_uniweight_1row4col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight_int8input_inplace(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups);

tinyengine_status_fp depthwise_conv_fp_kernelx_stride1_pad0 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp depthwise_conv_fp_kernel3_stride1_pad1(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp depthwise_conv_fp_kernel3_stride1_pad3(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp depthwise_conv_fp_kernel3_stride2_pad1 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp depthwise_conv_fp_kernelx_stride1_pad1 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp depthwise_conv_fp_kernelx_stride1_pad2 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp depthwise_conv_fp_kernelx_stride1_pad3 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp depthwise_conv_fp_kernel4_stride1_pad1_in4x4_out3x3_uniweight_1row1col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp depthwise_conv_fp_kernel4_stride1_pad1_dil2_in8x8_out4x4_uniweight_1row1col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp depthwise_conv_fp_kernel8_stride1_pad1_dil1_in8x8_out3x3_uniweight_1row1col_inplace(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status_fp pointwise_conv_fp_1row16col_10inputdepth(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row4col_10inputdepth(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row16col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8w_partialCH(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row8col_10inputdepth_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row8col_10inputdepth(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_1row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_2row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row16col_IOHW_int8w_partialCH(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row16col_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row8col_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8w_partialCH(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row4col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row2col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp pointwise_conv_fp_4row1col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel3_stride1_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel3_stride1_inpad1_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel3_stride2_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel3_stride2_inpad1_outpad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel5_stride1_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel5_stride1_inpad2_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel5_stride2_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel5_stride2_inpad2_outpad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel7_stride1_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel7_stride1_inpad3_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel7_stride2_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernel7_stride2_inpad3_outpad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_conv_fp_kernelx_stride1_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride1_inpad1_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride1_inpad2_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride1_inpad1_outpad0_revised(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad0_revised(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad1_revised(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride1_inpad2_outpad0_revised(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad0_revised(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_revised(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_revised(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad1_revised(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_inw3_revised(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride1_inpad1_outpad0_revised_IOHW_int8w(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad0_revised_IOHW_int8w(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad1_revised_IOHW_int8w(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride1_inpad2_outpad0_revised_IOHW_int8w(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad0_revised_IOHW_int8w(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_revised_IOHW_int8w(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_revised_IOHW_int8w(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised_IOHW_int8w(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad1_revised_IOHW_int8w(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride1_inpad1_outpad0_revised_IOHW_int8w_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad0_revised_IOHW_int8w_partialCH(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel3_stride2_inpad1_outpad1_revised_IOHW_int8w_partialCH(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride1_inpad2_outpad0_revised_IOHW_int8w_partialCH(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad0_revised_IOHW_int8w_partialCH(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel5_stride2_inpad2_outpad1_revised_IOHW_int8w_partialCH(float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_revised_IOHW_int8w_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised_IOHW_int8w_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad1_revised_IOHW_int8w_partialCH(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const float output_activation_min, const float output_activation_max,
		float* im2col_data, const uint16_t batches, const int pad_value);

tinyengine_status_fp group_conv_fp(const Conv_Params params, int group,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t _input_depth,
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t _output_depth,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp group_conv_fp_inplace(const Conv_Params params, int group,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t _input_depth,
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data,
		int8_t* weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t _output_depth,
		float* im2col_data, const uint16_t batches,
		const float* scales, const float learning_rate);

tinyengine_status_fp group_conv_fp_int8input_inplace(const Conv_Params params, int group,
		const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t _input_depth,
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data, 
		int8_t* weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t _output_depth,
		float* im2col_data, const uint16_t batches,
		const float* scales, const float learning_rate);

tinyengine_status_fp TFLite_DepthwiseConv(const Depthwise_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_DepthwiseConv_int8_PerChannel(const Depthwise_Params params,
		const int32_t* output_multiplier, const int32_t* output_shift,
		const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const int8_t* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		const uint16_t batches);

tinyengine_status_fp TFLite_DepthwiseConv_int8_PerChannel_partialCH(const Depthwise_Params params,
		const int32_t* output_multiplier, const int32_t* output_shift,
		const int8_t* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const int8_t* filter_sram,const int8_t* filter_flash,const uint16_t first_k_channel, const uint16_t filter_height, const uint16_t filter_width,
		const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		const uint16_t batches);

tinyengine_status_fp TFLite_TransposeConv(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_TransposeConv_IOHW(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data,
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_TransposeConv_IOHW_int8w(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const int8_t* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data,
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_TransposeConv_IOHW_int8w_partialCH(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const int8_t* filter_sram,const int8_t* filter_flash,const uint16_t first_k_channel, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data,
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_TransposeDepthwiseConv(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const float* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_TransposeDepthwiseConv_int8w(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const int8_t* filter_data, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data,
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp TFLite_TransposeDepthwiseConv_int8w_partialCH(const Conv_Params params,
		const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		const int8_t* filter_sram,const int8_t* filter_flash,const uint16_t first_k_channel, const uint16_t filter_height, const uint16_t filter_width,
		const float* bias_data,
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth,
		float* im2col_data, const uint16_t batches);

tinyengine_status_fp LogSoftmax(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth);

tinyengine_status_fp nll_loss(const float* input_data, const uint16_t input_dim, const uint16_t input_depth, 
		const float* target, const uint16_t target_size, float* output_data);

tinyengine_status_fp transpose_2Dto2D(const float* input, const uint16_t input_x, const uint16_t input_y, float* output);

tinyengine_status_fp transpose_3Dto3D(const float* input, const uint16_t input_h, const uint16_t input_w,
		const uint16_t input_c, float* output);

tinyengine_status_fp strided_slice_3Dto3D(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
		const uint16_t* begin, const uint16_t* end, const uint16_t* stride, 
		float* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c);

tinyengine_status_fp strided_slice_4Dto4D(const float* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c1, const uint16_t input_c2,
		const uint16_t* begin, const uint16_t* end, const uint16_t* stride, 
		float* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c1, const uint16_t output_c2);

tinyengine_status_fp reshape_3dto1d(const float* input, const uint16_t h, const uint16_t w, const uint16_t c, float* output);

tinyengine_status_fp permute3D_dim120(float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth,
		float* output_data);

tinyengine_status_fp permute4D_dim3012(float* input_data, const uint16_t d1, const uint16_t d2, const uint16_t d3, const uint16_t d4,
                       float* sbuf);

tinyengine_status_fp permute_groupconv_out(float* input_data, const uint16_t d1, const uint16_t d2, const uint16_t d3, const uint16_t input_c, const uint16_t output_c,
                       float* output_data);

tinyengine_status_fp sum_4D_exclude(const float* input_data, const uint16_t d1, const uint16_t d2,
                      const uint16_t d3, const uint16_t d4, const uint16_t axis, float* output_data);
