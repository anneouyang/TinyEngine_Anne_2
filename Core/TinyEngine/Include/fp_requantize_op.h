/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.edu
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#ifndef TINYENGINE_INCLUDE_FP_REQUANTIZE_OP_H_
#define TINYENGINE_INCLUDE_FP_REQUANTIZE_OP_H_

tinyengine_status convolve_1x1_s8_ch8_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch16_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch24_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_ch48_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status group_conv_s8_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch, const uint16_t group,
		const q7_t *kernel, const int16_t kernel_x, const int16_t kernel_y,
		const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

tinyengine_status convolve_1x1_s8_fpreq_bitmask(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, q7_t *mask, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf);

q7_t* mat_mult_kernel_s8_s16_reordered_fpreq(const q7_t *input_a,
		const q15_t *input_b, const uint16_t output_ch, const float *scales,
		const int32_t out_offset, const int16_t activation_min,
		const int16_t activation_max, const uint16_t num_col_a,
		const int32_t *const output_bias, q7_t *out_0);

q7_t* mat_mult_kernel_s8_s16_reordered_fpreq_group(const q7_t *input_a,
		const q15_t *input_b, const uint16_t output_ch, uint16_t output_group_ch, const float *scales,
		const int32_t out_offset, const int16_t activation_min,
		const int16_t activation_max, const uint16_t num_col_a,
		const int32_t *const output_bias, q7_t *out_0);


q7_t* mat_mult_kernel_s8_s16_reordered_ch8_fpreq(const q7_t *input_a,
		const q15_t *input_b, const uint16_t output_ch, const float *scales,
		const int32_t out_offset, const int16_t activation_min,
		const int16_t activation_max, const uint16_t num_col_a,
		const int32_t *const output_bias, q7_t *out_0);

q7_t* mat_mult_kernel_s8_s16_reordered_ch16_fpreq(const q7_t *input_a,
		const q15_t *input_b, const uint16_t output_ch, const float *scales,
		const int32_t out_offset, const int16_t activation_min,
		const int16_t activation_max, const uint16_t num_col_a,
		const int32_t *const output_bias, q7_t *out_0);

q7_t* mat_mult_kernel_s8_s16_reordered_ch24_fpreq(const q7_t *input_a,
		const q15_t *input_b, const uint16_t output_ch, const float *scales,
		const int32_t out_offset, const int16_t activation_min,
		const int16_t activation_max, const uint16_t num_col_a,
		const int32_t *const output_bias, q7_t *out_0);

q7_t* mat_mult_kernel_s8_s16_reordered_ch48_fpreq(const q7_t *input_a,
		const q15_t *input_b, const uint16_t output_ch, const float *scales,
		const int32_t out_offset, const int16_t activation_min,
		const int16_t activation_max, const uint16_t num_col_a,
		const int32_t *const output_bias, q7_t *out_0);

#endif /* TINYENGINE_INCLUDE_FP_REQUANTIZE_OP_H_ */