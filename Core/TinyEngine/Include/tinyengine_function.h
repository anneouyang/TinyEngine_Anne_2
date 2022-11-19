/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.ed
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

typedef int8_t q7_t;
typedef uint8_t q8_t;
typedef int16_t q15_t;
typedef uint16_t q16_t;
typedef int32_t q31_t;
typedef uint32_t q32_t;

typedef enum {
	STATE_SUCCESS = 0, /* No error */
	PARAM_NO_SUPPORT = 1, /* Unsupported parameters */
} tinyengine_status;

typedef struct add_params {
	int input_h, input_w, input_c, left_shift;
	int input1_offset, input1_multiplier, input1_shift;
	int input2_offset, input2_multiplier, input2_shift;
	int output_offset, output_multiplier, output_shift;
	int quantized_activation_max, quantized_activation_min;

} ADD_params;

#define TN_MAX(A,B) ((A) > (B) ? (A) : (B))
#define TN_MIN(A,B) ((A) < (B) ? (A) : (B))

// bit assignment and check
#define BIT_SET(a,b) ((a) |= (1ULL<<(b)))
#define BIT_CLEAR(a,b) ((a) &= ~(1ULL<<(b)))
#define BIT_FLIP(a,b) ((a) ^= (1ULL<<(b)))
#define BIT_CHECK(a,b) (!!((a) & (1ULL<<(b))))        // '!!' to make sure this returns 0 or 1

#define BITMASK_SET(x, mask) ((x) |= (mask))
#define BITMASK_CLEAR(x, mask) ((x) &= (~(mask)))
#define BITMASK_FLIP(x, mask) ((x) ^= (mask))
#define BITMASK_CHECK_ALL(x, mask) (!(~(x) & (mask)))
#define BITMASK_CHECK_ANY(x, mask) ((x) & (mask))

tinyengine_status convolve_1x1_s8(const q7_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const q7_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch8(const q7_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const q7_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch16(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch24(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch48(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_s8_kernel3_inputch3_stride2_pad1(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t output_offset,
		const int32_t input_offset, const int32_t output_activation_min,
		const int32_t output_activation_max, q7_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf, q15_t *kbuf,
		q7_t pad_value);

tinyengine_status add(int size, ADD_params *params, const int8_t *input1_data,
		const int8_t *input2_data, int8_t *output_data);

tinyengine_status avg_pooling(const q7_t *input, const uint16_t input_h,
		const uint16_t input_w, const uint16_t input_c, const uint16_t sample_h,
		const uint16_t sample_w, const uint16_t output_h,
		const uint16_t output_w, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output);

tinyengine_status fully_connected_fp(const float *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch,
		const uint16_t output_ch, const float *bias, const float *weights,
		float *output);

tinyengine_status statble_softmax_inplace(float *input, const uint16_t length);

tinyengine_status mat_mul_fp(const float *matA, const uint16_t matA_row,
		const uint16_t matA_col, const float *matB, const uint16_t matB_col,
		float *output);

tinyengine_status convolve_s8_kernel3_inputch3_stride2_pad1_fpreq(
		const q7_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, const q7_t *kernel, const int32_t *bias,
		const float *scales, const int32_t output_offset,
		const int32_t input_offset, const int32_t output_activation_min,
		const int32_t output_activation_max, q7_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf, q15_t *kbuf,
		q7_t pad_value);

tinyengine_status add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data);

tinyengine_status add_fpreq_mask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask);

tinyengine_status add_fpreq_bitmask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask);

tinyengine_status convolve_1x1_s8_fpreq_mask_partialCH(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel_sram, const q7_t *kernel_flash, const uint16_t first_k_channel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, q7_t *mask, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);


/****** OPs for INT8 BP ******/
tinyengine_status strided_slice_3Dto3D_int8(const q7_t* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
		const uint16_t* begin, const uint16_t* end, const uint16_t* stride, 
		q7_t* output, const uint16_t output_h, const uint16_t output_w, const uint16_t output_c);

tinyengine_status strided_slice_4Dto4D_int8(const q7_t* input, const uint16_t inn, const uint16_t inc, const uint16_t inh, const uint16_t inw,
		const uint16_t* begin, const uint16_t* end, const uint16_t* stride,
		q7_t* output, const uint16_t on, const uint16_t oc, const uint16_t oh, const uint16_t ow);

tinyengine_status sum_2D_int8(const q7_t* input_data, const uint16_t matA_row,
		const uint16_t matA_col, const uint16_t axis, q31_t* output_data);

tinyengine_status sum_3D_int8(const q7_t* input_data, const uint16_t input_w, const uint16_t input_h,
		const uint16_t input_c, const uint16_t axis, q31_t* output_data);

tinyengine_status sum_4D_exclude_int8(const q7_t* input_data, const uint16_t d1, const uint16_t d2,
		const uint16_t d3, const uint16_t d4, const uint16_t axis, q31_t* output_data);

tinyengine_status where_int8(const bool* inMask, const uint16_t size, const q7_t* input1_data,
		const q7_t* input2_data, q7_t* output_data);

tinyengine_status where_zeros_int8(const bool* inMask, const uint16_t size, const q7_t* input1_data, q7_t* output_data);

tinyengine_status where_zeros_int8_inplace(const bool* inMask, const uint16_t size, q7_t* input1_data);

tinyengine_status where_zeros_int8_inplace_bit(const unsigned char* inMask, const uint16_t size, q7_t* input1_data);

tinyengine_status group_pointwise_conv_in1x1_out1x1_1row10col_uniweight_int8input_inplace(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		int16_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status group_conv_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_int8input_int8weight_inplace_revised(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		q7_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status group_conv_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_int8input_int8weight_inplace_revised(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		q7_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status group_conv_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_int8input_int8weight_inplace_revised(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		q7_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status group_conv_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_int8input_int8weight_inplace_revised(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		q7_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
		const float* scales, const float learning_rate);

tinyengine_status pointwise_conv_1row10col_10inputdepth_IOHW_int8w(const int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int16_t* im2col_data, q31_t* norm_data, const uint16_t batches);

tinyengine_status pointwise_conv_4row4col_IOHW_int8input_int8w(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		int16_t* im2col_data, q31_t* norm_data, const uint16_t batches);

tinyengine_status pointwise_conv_4row4col_IOHW_int8input_int8w_SIMD(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		int16_t* im2col_data, q31_t* norm_data, const uint16_t batches);

tinyengine_status pointwise_conv_4row4col_IOHW_int8w_SIMD(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const q31_t* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		int16_t* im2col_data, const uint16_t batches);

tinyengine_status pointwise_conv_4row4col_IOHW_int8input_int8w_partialCH(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_sram, const q7_t* filter_flash, const uint16_t first_k_channel, const q31_t* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		int16_t* im2col_data, q31_t* norm_data, const uint16_t batches);

tinyengine_status pointwise_conv_4row4col_4innercol_IOHW_int8input_int8w_partialCH(const q7_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_sram, const q7_t* filter_flash, const uint16_t first_k_channel, const q31_t* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const q31_t output_activation_min, const q31_t output_activation_max,
		int16_t* im2col_data, q31_t* norm_data, const uint16_t batches);

tinyengine_status transpose_depthwise_conv_kernel3_stride1_inpad1_outpad0_revised_IOHW_partialCH(int8_t* input_output_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel3_stride1_inpad1_outpad0_revised_IOHW(int8_t* input_output_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel3_stride2_inpad1_outpad0_revised_IOHW_partialCH(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel3_stride2_inpad1_outpad0_revised_IOHW(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel3_stride2_inpad1_outpad1_revised_IOHW_partialCH(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel3_stride2_inpad1_outpad1_revised_IOHW(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel5_stride1_inpad2_outpad0_revised_IOHW_partialCH(int8_t* input_output_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel5_stride1_inpad2_outpad0_revised_IOHW(int8_t* input_output_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel5_stride2_inpad2_outpad0_revised_IOHW_partialCH(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel5_stride2_inpad2_outpad0_revised_IOHW(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel5_stride2_inpad2_outpad1_revised_IOHW_partialCH(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel5_stride2_inpad2_outpad1_revised_IOHW(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel7_stride1_inpad3_outpad0_revised_IOHW_partialCH(int8_t* input_output_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel7_stride1_inpad3_outpad0_revised_IOHW(int8_t* input_output_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel7_stride2_inpad3_outpad0_revised_IOHW_partialCH(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel7_stride2_inpad3_outpad0_revised_IOHW(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel7_stride2_inpad3_outpad1_revised_IOHW_partialCH(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

tinyengine_status transpose_depthwise_conv_kernel7_stride2_inpad3_outpad1_revised_IOHW(int8_t* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const int8_t* filter_data, const int32_t* bias_data, 
		int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value);

/* float input, int8 output ops */
tinyengine_status pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8output_int8w(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const float* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int16_t* im2col_data, float* norm_data, const uint16_t batches);

tinyengine_status pointwise_conv_fp_4row4col_IOHW_int8output_int8w(const float* input_data, 
		const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
		const q7_t* filter_data, const float* bias_data, 
		q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
		const int32_t output_activation_min, const int32_t output_activation_max,
		int16_t* im2col_data, float* norm_data, const uint16_t batches);


#include "genInclude.h"
#include "fp_requantize_op.h"
