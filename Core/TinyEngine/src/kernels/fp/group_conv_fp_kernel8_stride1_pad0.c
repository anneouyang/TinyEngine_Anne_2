/* ----------------------------------------------------------------------
 * Name: group_conv_fp_kernel8_stride1_pad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */
#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (8U)
#define DIM_KER_Y (8U)

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace_partialCH_half(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const uint16_t filter_depth, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const float* src_0;
  const float* src_1;
  const float* src_2;
  const float* src_3;

  int group, i, j;
  int output_depth_per_group = output_depth / groups;
  //int number_tile = output_depth_per_group / filter_depth;

  for (group = 0; group < groups - 3; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = *src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = *src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];

    const float* filter = filter_data;

    uint16_t col_count_div8 = filter_depth >> 3;
    int output_ch_count = 0;

    while (col_count_div8--) {
      float sum_0[8] = {};
      float sum_1[8] = {};
      float sum_2[8] = {};
      float sum_3[8] = {};

      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, filter_depth);
      filter++;

      /* Calculate outputs */
      /*
      output_weight_data[(output_ch_count * groups + group) * 2]           = sum_0[0];
      output_weight_data[(output_ch_count * groups + group) * 2 + 1]       = sum_0[0];
      output_weight_data[((output_ch_count + 1) * groups + group) * 2]     = sum_0[1];
      output_weight_data[((output_ch_count + 1) * groups + group) * 2 + 1] = sum_0[1];
      output_weight_data[((output_ch_count + 2) * groups + group) * 2]     = sum_0[2];
      output_weight_data[((output_ch_count + 2) * groups + group) * 2 + 1] = sum_0[2];
      output_weight_data[((output_ch_count + 3) * groups + group) * 2]     = sum_0[3];
      output_weight_data[((output_ch_count + 3) * groups + group) * 2 + 1] = sum_0[3];
      output_weight_data[((output_ch_count + 4) * groups + group) * 2]     = sum_0[4];
      output_weight_data[((output_ch_count + 4) * groups + group) * 2 + 1] = sum_0[4];
      output_weight_data[((output_ch_count + 5) * groups + group) * 2]     = sum_0[5];
      output_weight_data[((output_ch_count + 5) * groups + group) * 2 + 1] = sum_0[5];
      output_weight_data[((output_ch_count + 6) * groups + group) * 2]     = sum_0[6];
      output_weight_data[((output_ch_count + 6) * groups + group) * 2 + 1] = sum_0[6];
      output_weight_data[((output_ch_count + 7) * groups + group) * 2]     = sum_0[7];
      output_weight_data[((output_ch_count + 7) * groups + group) * 2 + 1] = sum_0[7];

      output_weight_data[(output_ch_count * groups + group + 1) * 2]           = sum_1[0];
      output_weight_data[(output_ch_count * groups + group + 1) * 2 + 1]       = sum_1[0];
      output_weight_data[((output_ch_count + 1) * groups + group + 1) * 2]     = sum_1[1];
      output_weight_data[((output_ch_count + 1) * groups + group + 1) * 2 + 1] = sum_1[1];
      output_weight_data[((output_ch_count + 2) * groups + group + 1) * 2]     = sum_1[2];
      output_weight_data[((output_ch_count + 2) * groups + group + 1) * 2 + 1] = sum_1[2];
      output_weight_data[((output_ch_count + 3) * groups + group + 1) * 2]     = sum_1[3];
      output_weight_data[((output_ch_count + 3) * groups + group + 1) * 2 + 1] = sum_1[3];
      output_weight_data[((output_ch_count + 4) * groups + group + 1) * 2]     = sum_1[4];
      output_weight_data[((output_ch_count + 4) * groups + group + 1) * 2 + 1] = sum_1[4];
      output_weight_data[((output_ch_count + 5) * groups + group + 1) * 2]     = sum_1[5];
      output_weight_data[((output_ch_count + 5) * groups + group + 1) * 2 + 1] = sum_1[5];
      output_weight_data[((output_ch_count + 6) * groups + group + 1) * 2]     = sum_1[6];
      output_weight_data[((output_ch_count + 6) * groups + group + 1) * 2 + 1] = sum_1[6];
      output_weight_data[((output_ch_count + 7) * groups + group + 1) * 2]     = sum_1[7];
      output_weight_data[((output_ch_count + 7) * groups + group + 1) * 2 + 1] = sum_1[7];

      output_weight_data[(output_ch_count * groups + group + 2) * 2]           = sum_2[0];
      output_weight_data[(output_ch_count * groups + group + 2) * 2 + 1]       = sum_2[0];
      output_weight_data[((output_ch_count + 1) * groups + group + 2) * 2]     = sum_2[1];
      output_weight_data[((output_ch_count + 1) * groups + group + 2) * 2 + 1] = sum_2[1];
      output_weight_data[((output_ch_count + 2) * groups + group + 2) * 2]     = sum_2[2];
      output_weight_data[((output_ch_count + 2) * groups + group + 2) * 2 + 1] = sum_2[2];
      output_weight_data[((output_ch_count + 3) * groups + group + 2) * 2]     = sum_2[3];
      output_weight_data[((output_ch_count + 3) * groups + group + 2) * 2 + 1] = sum_2[3];
      output_weight_data[((output_ch_count + 4) * groups + group + 2) * 2]     = sum_2[4];
      output_weight_data[((output_ch_count + 4) * groups + group + 2) * 2 + 1] = sum_2[4];
      output_weight_data[((output_ch_count + 5) * groups + group + 2) * 2]     = sum_2[5];
      output_weight_data[((output_ch_count + 5) * groups + group + 2) * 2 + 1] = sum_2[5];
      output_weight_data[((output_ch_count + 6) * groups + group + 2) * 2]     = sum_2[6];
      output_weight_data[((output_ch_count + 6) * groups + group + 2) * 2 + 1] = sum_2[6];
      output_weight_data[((output_ch_count + 7) * groups + group + 2) * 2]     = sum_2[7];
      output_weight_data[((output_ch_count + 7) * groups + group + 2) * 2 + 1] = sum_2[7];

      output_weight_data[(output_ch_count * groups + group + 3) * 2]           = sum_3[0];
      output_weight_data[(output_ch_count * groups + group + 3) * 2 + 1]       = sum_3[0];
      output_weight_data[((output_ch_count + 1) * groups + group + 3) * 2]     = sum_3[1];
      output_weight_data[((output_ch_count + 1) * groups + group + 3) * 2 + 1] = sum_3[1];
      output_weight_data[((output_ch_count + 2) * groups + group + 3) * 2]     = sum_3[2];
      output_weight_data[((output_ch_count + 2) * groups + group + 3) * 2 + 1] = sum_3[2];
      output_weight_data[((output_ch_count + 3) * groups + group + 3) * 2]     = sum_3[3];
      output_weight_data[((output_ch_count + 3) * groups + group + 3) * 2 + 1] = sum_3[3];
      output_weight_data[((output_ch_count + 4) * groups + group + 3) * 2]     = sum_3[4];
      output_weight_data[((output_ch_count + 4) * groups + group + 3) * 2 + 1] = sum_3[4];
      output_weight_data[((output_ch_count + 5) * groups + group + 3) * 2]     = sum_3[5];
      output_weight_data[((output_ch_count + 5) * groups + group + 3) * 2 + 1] = sum_3[5];
      output_weight_data[((output_ch_count + 6) * groups + group + 3) * 2]     = sum_3[6];
      output_weight_data[((output_ch_count + 6) * groups + group + 3) * 2 + 1] = sum_3[6];
      output_weight_data[((output_ch_count + 7) * groups + group + 3) * 2]     = sum_3[7];
      output_weight_data[((output_ch_count + 7) * groups + group + 3) * 2 + 1] = sum_3[7];
      */
      
      output_weight_data[(output_ch_count * groups + group) * 2]           -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0] * learning_rate;
      output_weight_data[(output_ch_count * groups + group) * 2 + 1]       -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0] * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group) * 2]     -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2]     * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2] * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group) * 2]     -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)]     * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)] * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group) * 2]     -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)]     * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)] * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group) * 2]     -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)]     * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)] * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group) * 2]     -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)]     * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)] * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group) * 2]     -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)]     * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)] * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group) * 2]     -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)]     * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group) * 2 + 1] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)] * learning_rate;

      output_weight_data[(output_ch_count * groups + group + 1) * 2]           -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0]           * learning_rate;
      output_weight_data[(output_ch_count * groups + group + 1) * 2 + 1]       -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0]       * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2]     * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2] * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)]     * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)] * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)]     * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)] * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)]     * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)] * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)]     * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)] * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)]     * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)] * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group + 1) * 2]     -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)]     * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group + 1) * 2 + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)] * learning_rate;

      output_weight_data[(output_ch_count * groups + group + 2) * 2]           -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0]           * learning_rate;
      output_weight_data[(output_ch_count * groups + group + 2) * 2 + 1]       -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0]       * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2]     * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2] * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)]     * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)] * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)]     * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)] * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)]     * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)] * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)]     * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)] * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)]     * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)] * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group + 2) * 2]     -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)]     * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group + 2) * 2 + 1] -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)] * learning_rate;

      output_weight_data[(output_ch_count * groups + group + 3) * 2]           -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0]           * learning_rate;
      output_weight_data[(output_ch_count * groups + group + 3) * 2 + 1]       -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count * 2 + (group/(groups/2)) ? 2 : 0]       * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2]     * learning_rate;
      output_weight_data[((output_ch_count + 1) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[(output_ch_count + 1) * 2] * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)]     * learning_rate;
      output_weight_data[((output_ch_count + 2) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[(output_ch_count + 2)] * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)]     * learning_rate;
      output_weight_data[((output_ch_count + 3) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[(output_ch_count + 3)] * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)]     * learning_rate;
      output_weight_data[((output_ch_count + 4) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[(output_ch_count + 4)] * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)]     * learning_rate;
      output_weight_data[((output_ch_count + 5) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[(output_ch_count + 5)] * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)]     * learning_rate;
      output_weight_data[((output_ch_count + 6) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[(output_ch_count + 6)] * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group + 3) * 2]     -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)]     * learning_rate;
      output_weight_data[((output_ch_count + 7) * groups + group + 3) * 2 + 1] -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[(output_ch_count + 7)] * learning_rate;

      output_ch_count += 8;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_int8input_inplace_revised(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 8) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    float* two_column_buffer_0 = im2col_data; float* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    float* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; float* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    float* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; float* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    float* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; float* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    const float* src_0 = filter_data++; const float* src_1 = filter_data++; const float* src_2 = filter_data++; const float* src_3 = filter_data++;
    const float* src_4 = filter_data++; const float* src_5 = filter_data++; const float* src_6 = filter_data++; const float* src_7 = filter_data++;

    for (i = 0; i < DIM_KER_X; i++) {
      for (j = 0; j < DIM_KER_Y; j++) {
        *two_column_buffer_0++ = *src_0; src_0 += output_depth_per_group;
        *two_column_buffer_1++ = *src_1; src_1 += output_depth_per_group;
        *two_column_buffer_2++ = *src_2; src_2 += output_depth_per_group;
        *two_column_buffer_3++ = *src_3; src_3 += output_depth_per_group;
        *two_column_buffer_4++ = *src_4; src_4 += output_depth_per_group;
        *two_column_buffer_5++ = *src_5; src_5 += output_depth_per_group;
        *two_column_buffer_6++ = *src_6; src_6 += output_depth_per_group;
        *two_column_buffer_7++ = *src_7; src_7 += output_depth_per_group;
      }
    }

    /* Setup output_weight_data */
    int8_t* out_0 = &output_weight_data[i_output_depth * groups]; int8_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    int8_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; int8_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    int8_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; int8_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    int8_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; int8_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];

    const int8_t* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
      const int8_t* src_8 = input++;
      const int8_t* src_9 = input++;
      const int8_t* src_10 = input++;
      const int8_t* src_11 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = (float)*src_8;
          src_8 += input_depth;
          *two_column_buffer_1++ = (float)*src_9;
          src_9 += input_depth;
          *two_column_buffer_2++ = (float)*src_10;
          src_10 += input_depth;
          *two_column_buffer_3++ = (float)*src_11;
          src_11 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];

      const float* filter = im2col_data;

      // We assume bias_data as zeros.
      float sum_0[8] = {};
      float sum_1[8] = {};
      float sum_2[8] = {};
      float sum_3[8] = {};
      
      /* Group Conv Computation */
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      assign_sum_to_group_output_4row8col(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, sum_0, sum_1, sum_2, sum_3, 
                                    output_activation_min, output_activation_max, scales, learning_rate, i_output_depth);
      out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4; out_4 += 4; out_5 += 4; out_6 += 4; out_7 += 4; 

      /*
      *out_0++ -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;

      *out_0++ -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;

      *out_0++ -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;

      *out_0++ -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      */
    }
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace_revised(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 8) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    float* two_column_buffer_0 = im2col_data; float* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    float* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; float* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    float* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; float* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    float* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; float* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    const float* src_0 = filter_data++; const float* src_1 = filter_data++; const float* src_2 = filter_data++; const float* src_3 = filter_data++;
    const float* src_4 = filter_data++; const float* src_5 = filter_data++; const float* src_6 = filter_data++; const float* src_7 = filter_data++;

    for (i = 0; i < DIM_KER_X; i++) {
      for (j = 0; j < DIM_KER_Y; j++) {
        *two_column_buffer_0++ = *src_0; src_0 += output_depth_per_group;
        *two_column_buffer_1++ = *src_1; src_1 += output_depth_per_group;
        *two_column_buffer_2++ = *src_2; src_2 += output_depth_per_group;
        *two_column_buffer_3++ = *src_3; src_3 += output_depth_per_group;
        *two_column_buffer_4++ = *src_4; src_4 += output_depth_per_group;
        *two_column_buffer_5++ = *src_5; src_5 += output_depth_per_group;
        *two_column_buffer_6++ = *src_6; src_6 += output_depth_per_group;
        *two_column_buffer_7++ = *src_7; src_7 += output_depth_per_group;
      }
    }

    /* Setup output_weight_data */
    int8_t* out_0 = &output_weight_data[i_output_depth * groups]; int8_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    int8_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; int8_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    int8_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; int8_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    int8_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; int8_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];

    const float* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
      src_0 = input++;
      src_1 = input++;
      src_2 = input++;
      src_3 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = *src_0;
          src_0 += input_depth;
          *two_column_buffer_1++ = *src_1;
          src_1 += input_depth;
          *two_column_buffer_2++ = *src_2;
          src_2 += input_depth;
          *two_column_buffer_3++ = *src_3;
          src_3 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];

      const float* filter = im2col_data;

      // We assume bias_data as zeros.
      float sum_0[8] = {};
      float sum_1[8] = {};
      float sum_2[8] = {};
      float sum_3[8] = {};
      
      /* Group Conv Computation */
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      assign_sum_to_group_output_4row8col(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, sum_0, sum_1, sum_2, sum_3, 
                                    output_activation_min, output_activation_max, scales, learning_rate, i_output_depth);
      out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4; out_4 += 4; out_5 += 4; out_6 += 4; out_7 += 4; 
    }
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_int8input_inplace(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const int8_t* src_0;
  const int8_t* src_1;
  const int8_t* src_2;
  const int8_t* src_3;

  int group, i, j;
  int output_depth_per_group = output_depth / groups;

  for (group = 0; group < groups - 3; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = (float)*src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = (float)*src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = (float)*src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = (float)*src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];

    const float* filter = filter_data;

    uint16_t col_count_div8 = output_depth_per_group >> 3;
    int output_ch_count = 0;

    while (col_count_div8--) {
      float sum_0[8] = {};
      float sum_1[8] = {};
      float sum_2[8] = {};
      float sum_3[8] = {};

      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 2] -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 2] -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 2] -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 2] -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 2] -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 2] -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 2] -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 2] -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 3] -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 3] -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 3] -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 3] -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 3] -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 3] -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 3] -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 3] -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_ch_count += 8;
    }
  }

  // TODO: For now, we assume leftover_groups is always 2.
  int leftover_groups = groups & 0x2;

  if (leftover_groups) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    src_0 = input_data++;
    src_1 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = (float)*src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = (float)*src_1;
        src_1 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];

    const float* filter = filter_data;

    uint16_t col_count_div8 = output_depth_per_group >> 3;
    int output_ch_count = 0;

    while (col_count_div8--) {
      float sum_0[8] = {};
      float sum_1[8] = {};

      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], input_0, input_1, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_ch_count += 8;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const float* src_0;
  const float* src_1;
  const float* src_2;
  const float* src_3;

  int group, i, j;
  int output_depth_per_group = output_depth / groups;

  for (group = 0; group < groups - 3; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = *src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = *src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];

    const float* filter = filter_data;

    uint16_t col_count_div8 = output_depth_per_group >> 3;
    int output_ch_count = 0;

    while (col_count_div8--) {
      float sum_0[8] = {};
      float sum_1[8] = {};
      float sum_2[8] = {};
      float sum_3[8] = {};

      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 2] -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 2] -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 2] -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 2] -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 2] -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 2] -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 2] -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 2] -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 3] -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 3] -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 3] -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 3] -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 3] -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 3] -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 3] -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 3] -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_ch_count += 8;
    }
  }

  // TODO: For now, we assume leftover_groups is always 2.
  int leftover_groups = groups & 0x2;

  if (leftover_groups) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    src_0 = input_data++;
    src_1 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];

    const float* filter = filter_data;

    uint16_t col_count_div8 = output_depth_per_group >> 3;
    int output_ch_count = 0;

    while (col_count_div8--) {
      float sum_0[8] = {};
      float sum_1[8] = {};

      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], input_0, input_1, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;

      output_ch_count += 8;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row8col_inplace_brutal(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const float* src_0;
  const float* src_1;
  const float* src_2;
  const float* src_3;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  /* Reshape filter_data and temporarily place it in im2col_data, and then replace the old filter_data with the new one. */
  float* filter = im2col_data;
  float* filter_data_start = filter_data;
  for (i = 0; i < output_depth_per_group; i++){
    for (j = 0; j < DIM_KER_Y * DIM_KER_X; j++) {
      *filter++ = *filter_data;
      filter_data += output_depth_per_group;
    }
    filter_data -= output_depth_per_group * DIM_KER_X * DIM_KER_Y - 1;
  }
  filter_data = filter_data_start;
  filter = im2col_data;
  for (i = 0; i < output_depth_per_group * DIM_KER_Y * DIM_KER_X; i++){
    *filter_data++ = *filter++;
  }

  int8_t* out_0 = output_weight_data;
  int8_t* out_1 = &output_weight_data[output_depth_per_group];
  int8_t* out_2 = &output_weight_data[output_depth_per_group * 2];
  int8_t* out_3 = &output_weight_data[output_depth_per_group * 3];

  for(group = 0; group < groups; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = *src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = *src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    filter = filter_data_start;

    uint16_t col_count_div8 = output_depth_per_group >> 3;
    int output_ch_count = 0;

    while (col_count_div8--) {
      float sum_0[8] = {};
      float sum_1[8] = {};
      float sum_2[8] = {};
      float sum_3[8] = {};

      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */
      *out_0++ += round(MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);

      *out_1++ += round(MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);

      *out_2++ += round(MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);

      *out_3++ += round(MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);

      output_ch_count += 8;
    }

    out_0 += output_depth_per_group * 3;
    out_1 += output_depth_per_group * 3;
    out_2 += output_depth_per_group * 3;
    out_3 += output_depth_per_group * 3;
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_int8input_inplace(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const int8_t* src_0;
  const int8_t* src_1;
  const int8_t* src_2;
  const int8_t* src_3;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  for (group = 0; group < groups; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = (float)*src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = (float)*src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = (float)*src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = (float)*src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];

    const float* filter = filter_data;

    uint16_t col_count_div16 = output_depth_per_group >> 4;
    int output_ch_count = 0;

    while (col_count_div16--) {
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};
      
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */      
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group] -= MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group] -= MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group] -= MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group] -= MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group] -= MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group] -= MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group] -= MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group] -= MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 1] -= MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 1] -= MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 1] -= MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 1] -= MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 1] -= MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 1] -= MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 1] -= MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 1] -= MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 2] -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 2] -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 2] -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 2] -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 2] -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 2] -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 2] -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 2] -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 2] -= MIN(MAX(sum_2[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 2] -= MIN(MAX(sum_2[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 2] -= MIN(MAX(sum_2[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 2] -= MIN(MAX(sum_2[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 2] -= MIN(MAX(sum_2[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 2] -= MIN(MAX(sum_2[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 2] -= MIN(MAX(sum_2[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 2] -= MIN(MAX(sum_2[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 3] -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 3] -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 3] -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 3] -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 3] -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 3] -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 3] -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 3] -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 3] -= MIN(MAX(sum_3[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 3] -= MIN(MAX(sum_3[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 3] -= MIN(MAX(sum_3[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 3] -= MIN(MAX(sum_3[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 3] -= MIN(MAX(sum_3[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 3] -= MIN(MAX(sum_3[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 3] -= MIN(MAX(sum_3[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 3] -= MIN(MAX(sum_3[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;
      

      output_ch_count += 16;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_inplace(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const float* src_0;
  const float* src_1;
  const float* src_2;
  const float* src_3;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  for (group = 0; group < groups; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = *src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = *src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];

    const float* filter = filter_data;

    uint16_t col_count_div16 = output_depth_per_group >> 4;
    int output_ch_count = 0;

    while (col_count_div16--) {
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};
      
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_4row_16col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */      
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group] -= MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group] -= MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group] -= MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group] -= MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group] -= MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group] -= MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group] -= MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group] -= MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 1] -= MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 1] -= MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 1] -= MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 1] -= MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 1] -= MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 1] -= MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 1] -= MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 1] -= MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 2] -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 2] -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 2] -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 2] -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 2] -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 2] -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 2] -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 2] -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 2] -= MIN(MAX(sum_2[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 2] -= MIN(MAX(sum_2[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 2] -= MIN(MAX(sum_2[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 2] -= MIN(MAX(sum_2[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 2] -= MIN(MAX(sum_2[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 2] -= MIN(MAX(sum_2[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 2] -= MIN(MAX(sum_2[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 2] -= MIN(MAX(sum_2[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 3] -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 3] -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 3] -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 3] -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 3] -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 3] -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 3] -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 3] -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 3] -= MIN(MAX(sum_3[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 3] -= MIN(MAX(sum_3[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 3] -= MIN(MAX(sum_3[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 3] -= MIN(MAX(sum_3[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 3] -= MIN(MAX(sum_3[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 3] -= MIN(MAX(sum_3[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 3] -= MIN(MAX(sum_3[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 3] -= MIN(MAX(sum_3[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;
      

      output_ch_count += 16;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_int8input_inplace_revised(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 16) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    float* two_column_buffer_0 = im2col_data; float* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    float* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; float* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    float* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; float* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    float* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; float* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    float* two_column_buffer_8 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8]; float* two_column_buffer_9 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
    float* two_column_buffer_10 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10]; float* two_column_buffer_11 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
    float* two_column_buffer_12 = &im2col_data[DIM_KER_X * DIM_KER_Y * 12]; float* two_column_buffer_13 = &im2col_data[DIM_KER_X * DIM_KER_Y * 13];
    float* two_column_buffer_14 = &im2col_data[DIM_KER_X * DIM_KER_Y * 14]; float* two_column_buffer_15 = &im2col_data[DIM_KER_X * DIM_KER_Y * 15];
    const float* src_0 = filter_data++; const float* src_1 = filter_data++; const float* src_2 = filter_data++; const float* src_3 = filter_data++;
    const float* src_4 = filter_data++; const float* src_5 = filter_data++; const float* src_6 = filter_data++; const float* src_7 = filter_data++;
    const float* src_8 = filter_data++; const float* src_9 = filter_data++; const float* src_10 = filter_data++; const float* src_11 = filter_data++;
    const float* src_12 = filter_data++; const float* src_13 = filter_data++; const float* src_14 = filter_data++; const float* src_15 = filter_data++;

    for (i = 0; i < DIM_KER_X; i++) {
      for (j = 0; j < DIM_KER_Y; j++) {
        *two_column_buffer_0++ = *src_0; src_0 += output_depth_per_group;
        *two_column_buffer_1++ = *src_1; src_1 += output_depth_per_group;
        *two_column_buffer_2++ = *src_2; src_2 += output_depth_per_group;
        *two_column_buffer_3++ = *src_3; src_3 += output_depth_per_group;
        *two_column_buffer_4++ = *src_4; src_4 += output_depth_per_group;
        *two_column_buffer_5++ = *src_5; src_5 += output_depth_per_group;
        *two_column_buffer_6++ = *src_6; src_6 += output_depth_per_group;
        *two_column_buffer_7++ = *src_7; src_7 += output_depth_per_group;
        *two_column_buffer_8++ = *src_8; src_8 += output_depth_per_group;
        *two_column_buffer_9++ = *src_9; src_9 += output_depth_per_group;
        *two_column_buffer_10++ = *src_10; src_10 += output_depth_per_group;
        *two_column_buffer_11++ = *src_11; src_11 += output_depth_per_group;
        *two_column_buffer_12++ = *src_12; src_12 += output_depth_per_group;
        *two_column_buffer_13++ = *src_13; src_13 += output_depth_per_group;
        *two_column_buffer_14++ = *src_14; src_14 += output_depth_per_group;
        *two_column_buffer_15++ = *src_15; src_15 += output_depth_per_group;
      }
    }

    /* Setup output_weight_data */
    int8_t* out_0 = &output_weight_data[i_output_depth * groups]; int8_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    int8_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; int8_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    int8_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; int8_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    int8_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; int8_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];
    int8_t* out_8 = &output_weight_data[(i_output_depth + 8) * groups]; int8_t* out_9 = &output_weight_data[(i_output_depth + 9) * groups];
    int8_t* out_10 = &output_weight_data[(i_output_depth + 10) * groups]; int8_t* out_11 = &output_weight_data[(i_output_depth + 11) * groups];
    int8_t* out_12 = &output_weight_data[(i_output_depth + 12) * groups]; int8_t* out_13 = &output_weight_data[(i_output_depth + 13) * groups];
    int8_t* out_14 = &output_weight_data[(i_output_depth + 14) * groups]; int8_t* out_15 = &output_weight_data[(i_output_depth + 15) * groups];

    const int8_t* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];
      const int8_t* src_16 = input++;
      const int8_t* src_17 = input++;
      const int8_t* src_18 = input++;
      const int8_t* src_19 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = (float)*src_16;
          src_16 += input_depth;
          *two_column_buffer_1++ = (float)*src_17;
          src_17 += input_depth;
          *two_column_buffer_2++ = (float)*src_18;
          src_18 += input_depth;
          *two_column_buffer_3++ = (float)*src_19;
          src_19 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];

      const float* filter = im2col_data;

      // We assume bias_data as zeros.
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};
      
      /* Group Conv Computation */
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      assign_sum_to_group_output_4row16col(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, 
                                    sum_0, sum_1, sum_2, sum_3, output_activation_min, output_activation_max, scales, learning_rate, i_output_depth);
      out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4; out_4 += 4; out_5 += 4; out_6 += 4; out_7 += 4; 
      out_8 += 4; out_9 += 4; out_10 += 4; out_11 += 4; out_12 += 4; out_13 += 4; out_14 += 4; out_15 += 4; 

      /*
      *out_0++ -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      *out_0++ -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      *out_0++ -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_2[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_2[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_2[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_2[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_2[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_2[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_2[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_2[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      *out_0++ -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_3[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_3[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_3[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_3[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_3[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_3[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_3[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_3[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;
      */
    }
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_inplace_revised(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 16) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    float* two_column_buffer_0 = im2col_data; float* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    float* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; float* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    float* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; float* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    float* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; float* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    float* two_column_buffer_8 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8]; float* two_column_buffer_9 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
    float* two_column_buffer_10 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10]; float* two_column_buffer_11 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
    float* two_column_buffer_12 = &im2col_data[DIM_KER_X * DIM_KER_Y * 12]; float* two_column_buffer_13 = &im2col_data[DIM_KER_X * DIM_KER_Y * 13];
    float* two_column_buffer_14 = &im2col_data[DIM_KER_X * DIM_KER_Y * 14]; float* two_column_buffer_15 = &im2col_data[DIM_KER_X * DIM_KER_Y * 15];
    const float* src_0 = filter_data++; const float* src_1 = filter_data++; const float* src_2 = filter_data++; const float* src_3 = filter_data++;
    const float* src_4 = filter_data++; const float* src_5 = filter_data++; const float* src_6 = filter_data++; const float* src_7 = filter_data++;
    const float* src_8 = filter_data++; const float* src_9 = filter_data++; const float* src_10 = filter_data++; const float* src_11 = filter_data++;
    const float* src_12 = filter_data++; const float* src_13 = filter_data++; const float* src_14 = filter_data++; const float* src_15 = filter_data++;

    for (i = 0; i < DIM_KER_X; i++) {
      for (j = 0; j < DIM_KER_Y; j++) {
        *two_column_buffer_0++ = *src_0; src_0 += output_depth_per_group;
        *two_column_buffer_1++ = *src_1; src_1 += output_depth_per_group;
        *two_column_buffer_2++ = *src_2; src_2 += output_depth_per_group;
        *two_column_buffer_3++ = *src_3; src_3 += output_depth_per_group;
        *two_column_buffer_4++ = *src_4; src_4 += output_depth_per_group;
        *two_column_buffer_5++ = *src_5; src_5 += output_depth_per_group;
        *two_column_buffer_6++ = *src_6; src_6 += output_depth_per_group;
        *two_column_buffer_7++ = *src_7; src_7 += output_depth_per_group;
        *two_column_buffer_8++ = *src_8; src_8 += output_depth_per_group;
        *two_column_buffer_9++ = *src_9; src_9 += output_depth_per_group;
        *two_column_buffer_10++ = *src_10; src_10 += output_depth_per_group;
        *two_column_buffer_11++ = *src_11; src_11 += output_depth_per_group;
        *two_column_buffer_12++ = *src_12; src_12 += output_depth_per_group;
        *two_column_buffer_13++ = *src_13; src_13 += output_depth_per_group;
        *two_column_buffer_14++ = *src_14; src_14 += output_depth_per_group;
        *two_column_buffer_15++ = *src_15; src_15 += output_depth_per_group;
      }
    }

    /* Setup output_weight_data */
    int8_t* out_0 = &output_weight_data[i_output_depth * groups]; int8_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    int8_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; int8_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    int8_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; int8_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    int8_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; int8_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];
    int8_t* out_8 = &output_weight_data[(i_output_depth + 8) * groups]; int8_t* out_9 = &output_weight_data[(i_output_depth + 9) * groups];
    int8_t* out_10 = &output_weight_data[(i_output_depth + 10) * groups]; int8_t* out_11 = &output_weight_data[(i_output_depth + 11) * groups];
    int8_t* out_12 = &output_weight_data[(i_output_depth + 12) * groups]; int8_t* out_13 = &output_weight_data[(i_output_depth + 13) * groups];
    int8_t* out_14 = &output_weight_data[(i_output_depth + 14) * groups]; int8_t* out_15 = &output_weight_data[(i_output_depth + 15) * groups];

    const float* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];
      src_0 = input++;
      src_1 = input++;
      src_2 = input++;
      src_3 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = *src_0;
          src_0 += input_depth;
          *two_column_buffer_1++ = *src_1;
          src_1 += input_depth;
          *two_column_buffer_2++ = *src_2;
          src_2 += input_depth;
          *two_column_buffer_3++ = *src_3;
          src_3 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];

      const float* filter = im2col_data;

      // We assume bias_data as zeros.
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};
      
      /* Group Conv Computation */
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      assign_sum_to_group_output_4row16col(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, 
                                    sum_0, sum_1, sum_2, sum_3, output_activation_min, output_activation_max, scales, learning_rate, i_output_depth);
      out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4; out_4 += 4; out_5 += 4; out_6 += 4; out_7 += 4; 
      out_8 += 4; out_9 += 4; out_10 += 4; out_11 += 4; out_12 += 4; out_13 += 4; out_14 += 4; out_15 += 4; 
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_4row16col_inplace_brutal(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  float* two_column_buffer_2;
  float* two_column_buffer_3;
  const float* src_0;
  const float* src_1;
  const float* src_2;
  const float* src_3;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  /* Reshape filter_data and temporarily place it in im2col_data, and then replace the old filter_data with the new one. */
  float* filter = im2col_data;
  float* filter_data_start = filter_data;
  for (i = 0; i < output_depth_per_group; i++){
    for (j = 0; j < DIM_KER_Y * DIM_KER_X; j++) {
      *filter++ = *filter_data;
      filter_data += output_depth_per_group;
    }
    filter_data -= output_depth_per_group * DIM_KER_X * DIM_KER_Y - 1;
  }
  filter_data = filter_data_start;
  filter = im2col_data;
  for (i = 0; i < output_depth_per_group * DIM_KER_Y * DIM_KER_X; i++){
    *filter_data++ = *filter++;
  }

  int8_t* out_0 = output_weight_data;
  int8_t* out_1 = &output_weight_data[output_depth_per_group];
  int8_t* out_2 = &output_weight_data[output_depth_per_group * 2];
  int8_t* out_3 = &output_weight_data[output_depth_per_group * 3];

  for(group = 0; group < groups; group += 4) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    src_0 = input_data++;
    src_1 = input_data++;
    src_2 = input_data++;
    src_3 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
        *two_column_buffer_2++ = *src_2;
        src_2 += input_depth;
        *two_column_buffer_3++ = *src_3;
        src_3 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2];
    const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    filter = filter_data_start;

    uint16_t col_count_div16 = output_depth_per_group >> 4;
    int output_ch_count = 0;

    while (col_count_div16--) {
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};

      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_16col_fp_uniweight(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */
      *out_0++ += round(MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate);

      *out_1++ += round(MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate);

      *out_2++ += round(MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate);
      *out_2++ += round(MIN(MAX(sum_2[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate);

      *out_3++ += round(MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate);
      *out_3++ += round(MIN(MAX(sum_3[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate);

      output_ch_count += 16;
    }

    out_0 += output_depth_per_group * 3;
    out_1 += output_depth_per_group * 3;
    out_2 += output_depth_per_group * 3;
    out_3 += output_depth_per_group * 3;
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_2row32col_int8input_inplace_revised(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, group, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 32) {
    float* two_column_buffer_0 = im2col_data; float* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    float* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; float* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    float* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; float* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    float* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; float* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    float* two_column_buffer_8 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8]; float* two_column_buffer_9 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
    float* two_column_buffer_10 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10]; float* two_column_buffer_11 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
    float* two_column_buffer_12 = &im2col_data[DIM_KER_X * DIM_KER_Y * 12]; float* two_column_buffer_13 = &im2col_data[DIM_KER_X * DIM_KER_Y * 13];
    float* two_column_buffer_14 = &im2col_data[DIM_KER_X * DIM_KER_Y * 14]; float* two_column_buffer_15 = &im2col_data[DIM_KER_X * DIM_KER_Y * 15];
    float* two_column_buffer_16 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16]; float* two_column_buffer_17 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
    float* two_column_buffer_18 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18]; float* two_column_buffer_19 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];
    float* two_column_buffer_20 = &im2col_data[DIM_KER_X * DIM_KER_Y * 20]; float* two_column_buffer_21 = &im2col_data[DIM_KER_X * DIM_KER_Y * 21];
    float* two_column_buffer_22 = &im2col_data[DIM_KER_X * DIM_KER_Y * 22]; float* two_column_buffer_23 = &im2col_data[DIM_KER_X * DIM_KER_Y * 23];
    float* two_column_buffer_24 = &im2col_data[DIM_KER_X * DIM_KER_Y * 24]; float* two_column_buffer_25 = &im2col_data[DIM_KER_X * DIM_KER_Y * 25];
    float* two_column_buffer_26 = &im2col_data[DIM_KER_X * DIM_KER_Y * 26]; float* two_column_buffer_27 = &im2col_data[DIM_KER_X * DIM_KER_Y * 27];
    float* two_column_buffer_28 = &im2col_data[DIM_KER_X * DIM_KER_Y * 28]; float* two_column_buffer_29 = &im2col_data[DIM_KER_X * DIM_KER_Y * 29];
    float* two_column_buffer_30 = &im2col_data[DIM_KER_X * DIM_KER_Y * 30]; float* two_column_buffer_31 = &im2col_data[DIM_KER_X * DIM_KER_Y * 31];

    const float* src_0 = filter_data++; const float* src_1 = filter_data++; const float* src_2 = filter_data++; const float* src_3 = filter_data++;
    const float* src_4 = filter_data++; const float* src_5 = filter_data++; const float* src_6 = filter_data++; const float* src_7 = filter_data++;
    const float* src_8 = filter_data++; const float* src_9 = filter_data++; const float* src_10 = filter_data++; const float* src_11 = filter_data++;
    const float* src_12 = filter_data++; const float* src_13 = filter_data++; const float* src_14 = filter_data++; const float* src_15 = filter_data++;
    const float* src_16 = filter_data++; const float* src_17 = filter_data++; const float* src_18 = filter_data++; const float* src_19 = filter_data++;
    const float* src_20 = filter_data++; const float* src_21 = filter_data++; const float* src_22 = filter_data++; const float* src_23 = filter_data++;
    const float* src_24 = filter_data++; const float* src_25 = filter_data++; const float* src_26 = filter_data++; const float* src_27 = filter_data++;
    const float* src_28 = filter_data++; const float* src_29 = filter_data++; const float* src_30 = filter_data++; const float* src_31 = filter_data++;

    for (i = 0; i < DIM_KER_X; i++) {
      for (j = 0; j < DIM_KER_Y; j++) {
        *two_column_buffer_0++ = *src_0; src_0 += output_depth_per_group;
        *two_column_buffer_1++ = *src_1; src_1 += output_depth_per_group;
        *two_column_buffer_2++ = *src_2; src_2 += output_depth_per_group;
        *two_column_buffer_3++ = *src_3; src_3 += output_depth_per_group;
        *two_column_buffer_4++ = *src_4; src_4 += output_depth_per_group;
        *two_column_buffer_5++ = *src_5; src_5 += output_depth_per_group;
        *two_column_buffer_6++ = *src_6; src_6 += output_depth_per_group;
        *two_column_buffer_7++ = *src_7; src_7 += output_depth_per_group;
        *two_column_buffer_8++ = *src_8; src_8 += output_depth_per_group;
        *two_column_buffer_9++ = *src_9; src_9 += output_depth_per_group;
        *two_column_buffer_10++ = *src_10; src_10 += output_depth_per_group;
        *two_column_buffer_11++ = *src_11; src_11 += output_depth_per_group;
        *two_column_buffer_12++ = *src_12; src_12 += output_depth_per_group;
        *two_column_buffer_13++ = *src_13; src_13 += output_depth_per_group;
        *two_column_buffer_14++ = *src_14; src_14 += output_depth_per_group;
        *two_column_buffer_15++ = *src_15; src_15 += output_depth_per_group;
        *two_column_buffer_16++ = *src_16; src_16 += output_depth_per_group;
        *two_column_buffer_17++ = *src_17; src_17 += output_depth_per_group;
        *two_column_buffer_18++ = *src_18; src_18 += output_depth_per_group;
        *two_column_buffer_19++ = *src_19; src_19 += output_depth_per_group;
        *two_column_buffer_20++ = *src_20; src_20 += output_depth_per_group;
        *two_column_buffer_21++ = *src_21; src_21 += output_depth_per_group;
        *two_column_buffer_22++ = *src_22; src_22 += output_depth_per_group;
        *two_column_buffer_23++ = *src_23; src_23 += output_depth_per_group;
        *two_column_buffer_24++ = *src_24; src_24 += output_depth_per_group;
        *two_column_buffer_25++ = *src_25; src_25 += output_depth_per_group;
        *two_column_buffer_26++ = *src_26; src_26 += output_depth_per_group;
        *two_column_buffer_27++ = *src_27; src_27 += output_depth_per_group;
        *two_column_buffer_28++ = *src_28; src_28 += output_depth_per_group;
        *two_column_buffer_29++ = *src_29; src_29 += output_depth_per_group;
        *two_column_buffer_30++ = *src_30; src_30 += output_depth_per_group;
        *two_column_buffer_31++ = *src_31; src_31 += output_depth_per_group;
      }
    }

    /* Setup output_weight_data */
    int8_t* out_0 = &output_weight_data[i_output_depth * groups]; int8_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    int8_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; int8_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    int8_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; int8_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    int8_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; int8_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];
    int8_t* out_8 = &output_weight_data[(i_output_depth + 8) * groups]; int8_t* out_9 = &output_weight_data[(i_output_depth + 9) * groups];
    int8_t* out_10 = &output_weight_data[(i_output_depth + 10) * groups]; int8_t* out_11 = &output_weight_data[(i_output_depth + 11) * groups];
    int8_t* out_12 = &output_weight_data[(i_output_depth + 12) * groups]; int8_t* out_13 = &output_weight_data[(i_output_depth + 13) * groups];
    int8_t* out_14 = &output_weight_data[(i_output_depth + 14) * groups]; int8_t* out_15 = &output_weight_data[(i_output_depth + 15) * groups];
    int8_t* out_16 = &output_weight_data[(i_output_depth + 16) * groups]; int8_t* out_17 = &output_weight_data[(i_output_depth + 17) * groups];
    int8_t* out_18 = &output_weight_data[(i_output_depth + 18) * groups]; int8_t* out_19 = &output_weight_data[(i_output_depth + 19) * groups];
    int8_t* out_20 = &output_weight_data[(i_output_depth + 20) * groups]; int8_t* out_21 = &output_weight_data[(i_output_depth + 21) * groups];
    int8_t* out_22 = &output_weight_data[(i_output_depth + 22) * groups]; int8_t* out_23 = &output_weight_data[(i_output_depth + 23) * groups];
    int8_t* out_24 = &output_weight_data[(i_output_depth + 24) * groups]; int8_t* out_25 = &output_weight_data[(i_output_depth + 25) * groups];
    int8_t* out_26 = &output_weight_data[(i_output_depth + 26) * groups]; int8_t* out_27 = &output_weight_data[(i_output_depth + 27) * groups];
    int8_t* out_28 = &output_weight_data[(i_output_depth + 28) * groups]; int8_t* out_29 = &output_weight_data[(i_output_depth + 29) * groups];
    int8_t* out_30 = &output_weight_data[(i_output_depth + 30) * groups]; int8_t* out_31 = &output_weight_data[(i_output_depth + 31) * groups];

    const int8_t* input = input_data;

    uint16_t group_cnt = groups >> 2;

    while (group_cnt--) {
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];
      const int8_t* src_16 = input++;
      const int8_t* src_17 = input++;
      const int8_t* src_18 = input++;
      const int8_t* src_19 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = (float)*src_16;
          src_16 += input_depth;
          *two_column_buffer_1++ = (float)*src_17;
          src_17 += input_depth;
          *two_column_buffer_2++ = (float)*src_18;
          src_18 += input_depth;
          *two_column_buffer_3++ = (float)*src_19;
          src_19 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      const float* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      const float* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];

      const float* filter = im2col_data;

      /*
      two_column_buffer_0 = im2col_data; two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
      two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
      two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
      two_column_buffer_8 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8]; two_column_buffer_9 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      two_column_buffer_10 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10]; two_column_buffer_11 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
      two_column_buffer_12 = &im2col_data[DIM_KER_X * DIM_KER_Y * 12]; two_column_buffer_13 = &im2col_data[DIM_KER_X * DIM_KER_Y * 13];
      two_column_buffer_14 = &im2col_data[DIM_KER_X * DIM_KER_Y * 14]; two_column_buffer_15 = &im2col_data[DIM_KER_X * DIM_KER_Y * 15];
      */

      //uint16_t row_count_div4 = input_depth >> 2;
      //int input_ch_count = 0;

      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};
      
      // TODO: use new functions for 2row group mac
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_4row_fp_uniweight_reuse_output_input(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      *out_0++ -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      *out_0++ -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      *out_0++ -= MIN(MAX(sum_2[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_2[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_2[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_2[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_2[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_2[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_2[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_2[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_2[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_2[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_2[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_2[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_2[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_2[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_2[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_2[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      *out_0++ -= MIN(MAX(sum_3[0], output_activation_min), output_activation_max) * scales[i_output_depth] * learning_rate;
      *out_1++ -= MIN(MAX(sum_3[1], output_activation_min), output_activation_max) * scales[i_output_depth + 1] * learning_rate;
      *out_2++ -= MIN(MAX(sum_3[2], output_activation_min), output_activation_max) * scales[i_output_depth + 2] * learning_rate;
      *out_3++ -= MIN(MAX(sum_3[3], output_activation_min), output_activation_max) * scales[i_output_depth + 3] * learning_rate;
      *out_4++ -= MIN(MAX(sum_3[4], output_activation_min), output_activation_max) * scales[i_output_depth + 4] * learning_rate;
      *out_5++ -= MIN(MAX(sum_3[5], output_activation_min), output_activation_max) * scales[i_output_depth + 5] * learning_rate;
      *out_6++ -= MIN(MAX(sum_3[6], output_activation_min), output_activation_max) * scales[i_output_depth + 6] * learning_rate;
      *out_7++ -= MIN(MAX(sum_3[7], output_activation_min), output_activation_max) * scales[i_output_depth + 7] * learning_rate;
      *out_8++ -= MIN(MAX(sum_3[8], output_activation_min), output_activation_max) * scales[i_output_depth + 8] * learning_rate;
      *out_9++ -= MIN(MAX(sum_3[9], output_activation_min), output_activation_max) * scales[i_output_depth + 9] * learning_rate;
      *out_10++ -= MIN(MAX(sum_3[10], output_activation_min), output_activation_max) * scales[i_output_depth + 10] * learning_rate;
      *out_11++ -= MIN(MAX(sum_3[11], output_activation_min), output_activation_max) * scales[i_output_depth + 11] * learning_rate;
      *out_12++ -= MIN(MAX(sum_3[12], output_activation_min), output_activation_max) * scales[i_output_depth + 12] * learning_rate;
      *out_13++ -= MIN(MAX(sum_3[13], output_activation_min), output_activation_max) * scales[i_output_depth + 13] * learning_rate;
      *out_14++ -= MIN(MAX(sum_3[14], output_activation_min), output_activation_max) * scales[i_output_depth + 14] * learning_rate;
      *out_15++ -= MIN(MAX(sum_3[15], output_activation_min), output_activation_max) * scales[i_output_depth + 15] * learning_rate;

      //output_ch_count += 4;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_2row32col_int8input_inplace(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  const int8_t* src_0;
  const int8_t* src_1;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  for(group = 0; group < groups; group += 2) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    src_0 = input_data++;
    src_1 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = (float)*src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = (float)*src_1;
        src_1 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    
    const float* filter = filter_data;

    uint16_t col_count_div32 = output_depth_per_group >> 5;
    int output_ch_count = 0;

    while (col_count_div32--) {
      float sum_0[32] = {};
      float sum_1[32] = {};

      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[16], &sum_1[16], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[17], &sum_1[17], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[18], &sum_1[18], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[19], &sum_1[19], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[20], &sum_1[20], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[21], &sum_1[21], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[22], &sum_1[22], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[23], &sum_1[23], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[24], &sum_1[24], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[25], &sum_1[25], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[26], &sum_1[26], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[27], &sum_1[27], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[28], &sum_1[28], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[29], &sum_1[29], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[30], &sum_1[30], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[31], &sum_1[31], input_0, input_1, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group] -= MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group] -= MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group] -= MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group] -= MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group] -= MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group] -= MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group] -= MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group] -= MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;
      output_weight_data[(output_ch_count + 16) * groups + group] -= MIN(MAX(sum_0[16], output_activation_min), output_activation_max) * scales[output_ch_count + 16] * learning_rate;
      output_weight_data[(output_ch_count + 17) * groups + group] -= MIN(MAX(sum_0[17], output_activation_min), output_activation_max) * scales[output_ch_count + 17] * learning_rate;
      output_weight_data[(output_ch_count + 18) * groups + group] -= MIN(MAX(sum_0[18], output_activation_min), output_activation_max) * scales[output_ch_count + 18] * learning_rate;
      output_weight_data[(output_ch_count + 19) * groups + group] -= MIN(MAX(sum_0[19], output_activation_min), output_activation_max) * scales[output_ch_count + 19] * learning_rate;
      output_weight_data[(output_ch_count + 20) * groups + group] -= MIN(MAX(sum_0[20], output_activation_min), output_activation_max) * scales[output_ch_count + 20] * learning_rate;
      output_weight_data[(output_ch_count + 21) * groups + group] -= MIN(MAX(sum_0[21], output_activation_min), output_activation_max) * scales[output_ch_count + 21] * learning_rate;
      output_weight_data[(output_ch_count + 22) * groups + group] -= MIN(MAX(sum_0[22], output_activation_min), output_activation_max) * scales[output_ch_count + 22] * learning_rate;
      output_weight_data[(output_ch_count + 23) * groups + group] -= MIN(MAX(sum_0[23], output_activation_min), output_activation_max) * scales[output_ch_count + 23] * learning_rate;
      output_weight_data[(output_ch_count + 24) * groups + group] -= MIN(MAX(sum_0[24], output_activation_min), output_activation_max) * scales[output_ch_count + 24] * learning_rate;
      output_weight_data[(output_ch_count + 25) * groups + group] -= MIN(MAX(sum_0[25], output_activation_min), output_activation_max) * scales[output_ch_count + 25] * learning_rate;
      output_weight_data[(output_ch_count + 26) * groups + group] -= MIN(MAX(sum_0[26], output_activation_min), output_activation_max) * scales[output_ch_count + 26] * learning_rate;
      output_weight_data[(output_ch_count + 27) * groups + group] -= MIN(MAX(sum_0[27], output_activation_min), output_activation_max) * scales[output_ch_count + 27] * learning_rate;
      output_weight_data[(output_ch_count + 28) * groups + group] -= MIN(MAX(sum_0[28], output_activation_min), output_activation_max) * scales[output_ch_count + 28] * learning_rate;
      output_weight_data[(output_ch_count + 29) * groups + group] -= MIN(MAX(sum_0[29], output_activation_min), output_activation_max) * scales[output_ch_count + 29] * learning_rate;
      output_weight_data[(output_ch_count + 30) * groups + group] -= MIN(MAX(sum_0[30], output_activation_min), output_activation_max) * scales[output_ch_count + 30] * learning_rate;
      output_weight_data[(output_ch_count + 31) * groups + group] -= MIN(MAX(sum_0[31], output_activation_min), output_activation_max) * scales[output_ch_count + 31] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 1] -= MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 1] -= MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 1] -= MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 1] -= MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 1] -= MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 1] -= MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 1] -= MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 1] -= MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;
      output_weight_data[(output_ch_count + 16) * groups + group + 1] -= MIN(MAX(sum_1[16], output_activation_min), output_activation_max) * scales[output_ch_count + 16] * learning_rate;
      output_weight_data[(output_ch_count + 17) * groups + group + 1] -= MIN(MAX(sum_1[17], output_activation_min), output_activation_max) * scales[output_ch_count + 17] * learning_rate;
      output_weight_data[(output_ch_count + 18) * groups + group + 1] -= MIN(MAX(sum_1[18], output_activation_min), output_activation_max) * scales[output_ch_count + 18] * learning_rate;
      output_weight_data[(output_ch_count + 19) * groups + group + 1] -= MIN(MAX(sum_1[19], output_activation_min), output_activation_max) * scales[output_ch_count + 19] * learning_rate;
      output_weight_data[(output_ch_count + 20) * groups + group + 1] -= MIN(MAX(sum_1[20], output_activation_min), output_activation_max) * scales[output_ch_count + 20] * learning_rate;
      output_weight_data[(output_ch_count + 21) * groups + group + 1] -= MIN(MAX(sum_1[21], output_activation_min), output_activation_max) * scales[output_ch_count + 21] * learning_rate;
      output_weight_data[(output_ch_count + 22) * groups + group + 1] -= MIN(MAX(sum_1[22], output_activation_min), output_activation_max) * scales[output_ch_count + 22] * learning_rate;
      output_weight_data[(output_ch_count + 23) * groups + group + 1] -= MIN(MAX(sum_1[23], output_activation_min), output_activation_max) * scales[output_ch_count + 23] * learning_rate;
      output_weight_data[(output_ch_count + 24) * groups + group + 1] -= MIN(MAX(sum_1[24], output_activation_min), output_activation_max) * scales[output_ch_count + 24] * learning_rate;
      output_weight_data[(output_ch_count + 25) * groups + group + 1] -= MIN(MAX(sum_1[25], output_activation_min), output_activation_max) * scales[output_ch_count + 25] * learning_rate;
      output_weight_data[(output_ch_count + 26) * groups + group + 1] -= MIN(MAX(sum_1[26], output_activation_min), output_activation_max) * scales[output_ch_count + 26] * learning_rate;
      output_weight_data[(output_ch_count + 27) * groups + group + 1] -= MIN(MAX(sum_1[27], output_activation_min), output_activation_max) * scales[output_ch_count + 27] * learning_rate;
      output_weight_data[(output_ch_count + 28) * groups + group + 1] -= MIN(MAX(sum_1[28], output_activation_min), output_activation_max) * scales[output_ch_count + 28] * learning_rate;
      output_weight_data[(output_ch_count + 29) * groups + group + 1] -= MIN(MAX(sum_1[29], output_activation_min), output_activation_max) * scales[output_ch_count + 29] * learning_rate;
      output_weight_data[(output_ch_count + 30) * groups + group + 1] -= MIN(MAX(sum_1[30], output_activation_min), output_activation_max) * scales[output_ch_count + 30] * learning_rate;
      output_weight_data[(output_ch_count + 31) * groups + group + 1] -= MIN(MAX(sum_1[31], output_activation_min), output_activation_max) * scales[output_ch_count + 31] * learning_rate;

      output_ch_count += 32;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_2row32col_inplace(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  const float* src_0;
  const float* src_1;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  for(group = 0; group < groups; group += 2) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    src_0 = input_data++;
    src_1 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    
    const float* filter = filter_data;

    uint16_t col_count_div32 = output_depth_per_group >> 5;
    int output_ch_count = 0;

    while (col_count_div32--) {
      float sum_0[32] = {};
      float sum_1[32] = {};

      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[16], &sum_1[16], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[17], &sum_1[17], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[18], &sum_1[18], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[19], &sum_1[19], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[20], &sum_1[20], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[21], &sum_1[21], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[22], &sum_1[22], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[23], &sum_1[23], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[24], &sum_1[24], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[25], &sum_1[25], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[26], &sum_1[26], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[27], &sum_1[27], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[28], &sum_1[28], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[29], &sum_1[29], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[30], &sum_1[30], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel8_2row_32col_fp_uniweight_IOHW(&sum_0[31], &sum_1[31], input_0, input_1, filter, output_depth_per_group);
      filter++;

      /* Calculate outputs */
      output_weight_data[output_ch_count * groups + group] -= MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group] -= MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group] -= MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group] -= MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group] -= MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group] -= MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group] -= MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group] -= MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group] -= MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group] -= MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group] -= MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group] -= MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group] -= MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group] -= MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group] -= MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group] -= MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;
      output_weight_data[(output_ch_count + 16) * groups + group] -= MIN(MAX(sum_0[16], output_activation_min), output_activation_max) * scales[output_ch_count + 16] * learning_rate;
      output_weight_data[(output_ch_count + 17) * groups + group] -= MIN(MAX(sum_0[17], output_activation_min), output_activation_max) * scales[output_ch_count + 17] * learning_rate;
      output_weight_data[(output_ch_count + 18) * groups + group] -= MIN(MAX(sum_0[18], output_activation_min), output_activation_max) * scales[output_ch_count + 18] * learning_rate;
      output_weight_data[(output_ch_count + 19) * groups + group] -= MIN(MAX(sum_0[19], output_activation_min), output_activation_max) * scales[output_ch_count + 19] * learning_rate;
      output_weight_data[(output_ch_count + 20) * groups + group] -= MIN(MAX(sum_0[20], output_activation_min), output_activation_max) * scales[output_ch_count + 20] * learning_rate;
      output_weight_data[(output_ch_count + 21) * groups + group] -= MIN(MAX(sum_0[21], output_activation_min), output_activation_max) * scales[output_ch_count + 21] * learning_rate;
      output_weight_data[(output_ch_count + 22) * groups + group] -= MIN(MAX(sum_0[22], output_activation_min), output_activation_max) * scales[output_ch_count + 22] * learning_rate;
      output_weight_data[(output_ch_count + 23) * groups + group] -= MIN(MAX(sum_0[23], output_activation_min), output_activation_max) * scales[output_ch_count + 23] * learning_rate;
      output_weight_data[(output_ch_count + 24) * groups + group] -= MIN(MAX(sum_0[24], output_activation_min), output_activation_max) * scales[output_ch_count + 24] * learning_rate;
      output_weight_data[(output_ch_count + 25) * groups + group] -= MIN(MAX(sum_0[25], output_activation_min), output_activation_max) * scales[output_ch_count + 25] * learning_rate;
      output_weight_data[(output_ch_count + 26) * groups + group] -= MIN(MAX(sum_0[26], output_activation_min), output_activation_max) * scales[output_ch_count + 26] * learning_rate;
      output_weight_data[(output_ch_count + 27) * groups + group] -= MIN(MAX(sum_0[27], output_activation_min), output_activation_max) * scales[output_ch_count + 27] * learning_rate;
      output_weight_data[(output_ch_count + 28) * groups + group] -= MIN(MAX(sum_0[28], output_activation_min), output_activation_max) * scales[output_ch_count + 28] * learning_rate;
      output_weight_data[(output_ch_count + 29) * groups + group] -= MIN(MAX(sum_0[29], output_activation_min), output_activation_max) * scales[output_ch_count + 29] * learning_rate;
      output_weight_data[(output_ch_count + 30) * groups + group] -= MIN(MAX(sum_0[30], output_activation_min), output_activation_max) * scales[output_ch_count + 30] * learning_rate;
      output_weight_data[(output_ch_count + 31) * groups + group] -= MIN(MAX(sum_0[31], output_activation_min), output_activation_max) * scales[output_ch_count + 31] * learning_rate;

      output_weight_data[output_ch_count * groups + group + 1] -= MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate;
      output_weight_data[(output_ch_count + 1) * groups + group + 1] -= MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate;
      output_weight_data[(output_ch_count + 2) * groups + group + 1] -= MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate;
      output_weight_data[(output_ch_count + 3) * groups + group + 1] -= MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate;
      output_weight_data[(output_ch_count + 4) * groups + group + 1] -= MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate;
      output_weight_data[(output_ch_count + 5) * groups + group + 1] -= MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate;
      output_weight_data[(output_ch_count + 6) * groups + group + 1] -= MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate;
      output_weight_data[(output_ch_count + 7) * groups + group + 1] -= MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate;
      output_weight_data[(output_ch_count + 8) * groups + group + 1] -= MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate;
      output_weight_data[(output_ch_count + 9) * groups + group + 1] -= MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate;
      output_weight_data[(output_ch_count + 10) * groups + group + 1] -= MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate;
      output_weight_data[(output_ch_count + 11) * groups + group + 1] -= MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate;
      output_weight_data[(output_ch_count + 12) * groups + group + 1] -= MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate;
      output_weight_data[(output_ch_count + 13) * groups + group + 1] -= MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate;
      output_weight_data[(output_ch_count + 14) * groups + group + 1] -= MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate;
      output_weight_data[(output_ch_count + 15) * groups + group + 1] -= MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate;
      output_weight_data[(output_ch_count + 16) * groups + group + 1] -= MIN(MAX(sum_1[16], output_activation_min), output_activation_max) * scales[output_ch_count + 16] * learning_rate;
      output_weight_data[(output_ch_count + 17) * groups + group + 1] -= MIN(MAX(sum_1[17], output_activation_min), output_activation_max) * scales[output_ch_count + 17] * learning_rate;
      output_weight_data[(output_ch_count + 18) * groups + group + 1] -= MIN(MAX(sum_1[18], output_activation_min), output_activation_max) * scales[output_ch_count + 18] * learning_rate;
      output_weight_data[(output_ch_count + 19) * groups + group + 1] -= MIN(MAX(sum_1[19], output_activation_min), output_activation_max) * scales[output_ch_count + 19] * learning_rate;
      output_weight_data[(output_ch_count + 20) * groups + group + 1] -= MIN(MAX(sum_1[20], output_activation_min), output_activation_max) * scales[output_ch_count + 20] * learning_rate;
      output_weight_data[(output_ch_count + 21) * groups + group + 1] -= MIN(MAX(sum_1[21], output_activation_min), output_activation_max) * scales[output_ch_count + 21] * learning_rate;
      output_weight_data[(output_ch_count + 22) * groups + group + 1] -= MIN(MAX(sum_1[22], output_activation_min), output_activation_max) * scales[output_ch_count + 22] * learning_rate;
      output_weight_data[(output_ch_count + 23) * groups + group + 1] -= MIN(MAX(sum_1[23], output_activation_min), output_activation_max) * scales[output_ch_count + 23] * learning_rate;
      output_weight_data[(output_ch_count + 24) * groups + group + 1] -= MIN(MAX(sum_1[24], output_activation_min), output_activation_max) * scales[output_ch_count + 24] * learning_rate;
      output_weight_data[(output_ch_count + 25) * groups + group + 1] -= MIN(MAX(sum_1[25], output_activation_min), output_activation_max) * scales[output_ch_count + 25] * learning_rate;
      output_weight_data[(output_ch_count + 26) * groups + group + 1] -= MIN(MAX(sum_1[26], output_activation_min), output_activation_max) * scales[output_ch_count + 26] * learning_rate;
      output_weight_data[(output_ch_count + 27) * groups + group + 1] -= MIN(MAX(sum_1[27], output_activation_min), output_activation_max) * scales[output_ch_count + 27] * learning_rate;
      output_weight_data[(output_ch_count + 28) * groups + group + 1] -= MIN(MAX(sum_1[28], output_activation_min), output_activation_max) * scales[output_ch_count + 28] * learning_rate;
      output_weight_data[(output_ch_count + 29) * groups + group + 1] -= MIN(MAX(sum_1[29], output_activation_min), output_activation_max) * scales[output_ch_count + 29] * learning_rate;
      output_weight_data[(output_ch_count + 30) * groups + group + 1] -= MIN(MAX(sum_1[30], output_activation_min), output_activation_max) * scales[output_ch_count + 30] * learning_rate;
      output_weight_data[(output_ch_count + 31) * groups + group + 1] -= MIN(MAX(sum_1[31], output_activation_min), output_activation_max) * scales[output_ch_count + 31] * learning_rate;

      output_ch_count += 32;
    }
  }
}

tinyengine_status_fp group_conv_fp_kernel8_stride1_pad0_in8x8_out1x1_uniweight_2row32col_inplace_brutal(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  float* two_column_buffer_0;
  float* two_column_buffer_1;
  const float* src_0;
  const float* src_1;

  int group, i , j;
  int output_depth_per_group = output_depth / groups;

  /* Reshape filter_data and temporarily place it in im2col_data, and then replace the old filter_data with the new one. */
  float* filter = im2col_data;
  float* filter_data_start = filter_data;
  for (i = 0; i < output_depth_per_group; i++){
    for (j = 0; j < DIM_KER_Y * DIM_KER_X; j++) {
      *filter++ = *filter_data;
      filter_data += output_depth_per_group;
    }
    filter_data -= output_depth_per_group * DIM_KER_X * DIM_KER_Y - 1;
  }
  filter_data = filter_data_start;
  filter = im2col_data;
  for (i = 0; i < output_depth_per_group * DIM_KER_Y * DIM_KER_X; i++){
    *filter_data++ = *filter++;
  }

  int8_t* out_0 = output_weight_data;
  int8_t* out_1 = &output_weight_data[output_depth_per_group];

  for(group = 0; group < groups; group += 2) {
    two_column_buffer_0 = im2col_data;
    two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    src_0 = input_data++;
    src_1 = input_data++;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer_0++ = *src_0;
        src_0 += input_depth;
        *two_column_buffer_1++ = *src_1;
        src_1 += input_depth;
      }
    }

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const float* input_0 = im2col_data;
    const float* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    filter = filter_data_start;

    uint16_t col_count_div32 = output_depth_per_group >> 5;
    int output_ch_count = 0;

    while (col_count_div32--) {
      float sum_0[32] = {};
      float sum_1[32] = {};

      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[0], &sum_1[0], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[1], &sum_1[1], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[2], &sum_1[2], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[3], &sum_1[3], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[4], &sum_1[4], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[5], &sum_1[5], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[6], &sum_1[6], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[7], &sum_1[7], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[8], &sum_1[8], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[9], &sum_1[9], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[10], &sum_1[10], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[11], &sum_1[11], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[12], &sum_1[12], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[13], &sum_1[13], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[14], &sum_1[14], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[15], &sum_1[15], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[16], &sum_1[16], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[17], &sum_1[17], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[18], &sum_1[18], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[19], &sum_1[19], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[20], &sum_1[20], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[21], &sum_1[21], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[22], &sum_1[22], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[23], &sum_1[23], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[24], &sum_1[24], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[25], &sum_1[25], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[26], &sum_1[26], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[27], &sum_1[27], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[28], &sum_1[28], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[29], &sum_1[29], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[30], &sum_1[30], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel8_2row_32col_fp_uniweight(&sum_0[31], &sum_1[31], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */
      *out_0++ += round(MIN(MAX(sum_0[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[16], output_activation_min), output_activation_max) * scales[output_ch_count + 16] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[17], output_activation_min), output_activation_max) * scales[output_ch_count + 17] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[18], output_activation_min), output_activation_max) * scales[output_ch_count + 18] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[19], output_activation_min), output_activation_max) * scales[output_ch_count + 19] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[20], output_activation_min), output_activation_max) * scales[output_ch_count + 20] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[21], output_activation_min), output_activation_max) * scales[output_ch_count + 21] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[22], output_activation_min), output_activation_max) * scales[output_ch_count + 22] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[23], output_activation_min), output_activation_max) * scales[output_ch_count + 23] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[24], output_activation_min), output_activation_max) * scales[output_ch_count + 24] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[25], output_activation_min), output_activation_max) * scales[output_ch_count + 25] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[26], output_activation_min), output_activation_max) * scales[output_ch_count + 26] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[27], output_activation_min), output_activation_max) * scales[output_ch_count + 27] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[28], output_activation_min), output_activation_max) * scales[output_ch_count + 28] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[29], output_activation_min), output_activation_max) * scales[output_ch_count + 29] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[30], output_activation_min), output_activation_max) * scales[output_ch_count + 30] * learning_rate);
      *out_0++ += round(MIN(MAX(sum_0[31], output_activation_min), output_activation_max) * scales[output_ch_count + 31] * learning_rate);

      *out_1++ += round(MIN(MAX(sum_1[0], output_activation_min), output_activation_max) * scales[output_ch_count] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[1], output_activation_min), output_activation_max) * scales[output_ch_count + 1] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[2], output_activation_min), output_activation_max) * scales[output_ch_count + 2] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[3], output_activation_min), output_activation_max) * scales[output_ch_count + 3] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[4], output_activation_min), output_activation_max) * scales[output_ch_count + 4] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[5], output_activation_min), output_activation_max) * scales[output_ch_count + 5] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[6], output_activation_min), output_activation_max) * scales[output_ch_count + 6] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[7], output_activation_min), output_activation_max) * scales[output_ch_count + 7] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[8], output_activation_min), output_activation_max) * scales[output_ch_count + 8] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[9], output_activation_min), output_activation_max) * scales[output_ch_count + 9] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[10], output_activation_min), output_activation_max) * scales[output_ch_count + 10] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[11], output_activation_min), output_activation_max) * scales[output_ch_count + 11] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[12], output_activation_min), output_activation_max) * scales[output_ch_count + 12] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[13], output_activation_min), output_activation_max) * scales[output_ch_count + 13] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[14], output_activation_min), output_activation_max) * scales[output_ch_count + 14] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[15], output_activation_min), output_activation_max) * scales[output_ch_count + 15] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[16], output_activation_min), output_activation_max) * scales[output_ch_count + 16] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[17], output_activation_min), output_activation_max) * scales[output_ch_count + 17] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[18], output_activation_min), output_activation_max) * scales[output_ch_count + 18] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[19], output_activation_min), output_activation_max) * scales[output_ch_count + 19] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[20], output_activation_min), output_activation_max) * scales[output_ch_count + 20] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[21], output_activation_min), output_activation_max) * scales[output_ch_count + 21] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[22], output_activation_min), output_activation_max) * scales[output_ch_count + 22] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[23], output_activation_min), output_activation_max) * scales[output_ch_count + 23] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[24], output_activation_min), output_activation_max) * scales[output_ch_count + 24] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[25], output_activation_min), output_activation_max) * scales[output_ch_count + 25] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[26], output_activation_min), output_activation_max) * scales[output_ch_count + 26] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[27], output_activation_min), output_activation_max) * scales[output_ch_count + 27] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[28], output_activation_min), output_activation_max) * scales[output_ch_count + 28] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[29], output_activation_min), output_activation_max) * scales[output_ch_count + 29] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[30], output_activation_min), output_activation_max) * scales[output_ch_count + 30] * learning_rate);
      *out_1++ += round(MIN(MAX(sum_1[31], output_activation_min), output_activation_max) * scales[output_ch_count + 31] * learning_rate);

      output_ch_count += 32;
    }

    out_0 += output_depth_per_group;
    out_1 += output_depth_per_group;
  }
}
