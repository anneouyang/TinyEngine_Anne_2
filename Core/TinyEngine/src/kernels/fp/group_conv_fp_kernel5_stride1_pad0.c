/* ----------------------------------------------------------------------
 * Name: group_conv_fp_kernel5_stride1_pad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (5U)
#define DIM_KER_Y (5U)

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row8col_int8input_inplace(const int8_t* input_data, 
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

      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
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
}

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row8col_inplace(const float* input_data, 
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

      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
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
}

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row16col_int8input_inplace(const int8_t* input_data, 
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

    uint16_t col_count_div16 = output_depth_per_group >> 4;
    int output_ch_count = 0;

    while (col_count_div16--) {
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};

      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
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

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row16col_inplace(const float* input_data, 
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

    uint16_t col_count_div16 = output_depth_per_group >> 4;
    int output_ch_count = 0;

    while (col_count_div16--) {
      float sum_0[16] = {};
      float sum_1[16] = {};
      float sum_2[16] = {};
      float sum_3[16] = {};

      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_4row_16col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter, output_depth_per_group);
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

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_4row16col_inplace_brutal(const float* input_data, 
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

  for(group = 0; group < groups - 3; group += 4) {
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

      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_4row_16col_fp_uniweight(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
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

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_2row32col_int8input_inplace(const int8_t* input_data, 
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

  for (group = 0; group < groups - 1; group += 2) {
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

      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[16], &sum_1[16], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[17], &sum_1[17], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[18], &sum_1[18], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[19], &sum_1[19], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[20], &sum_1[20], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[21], &sum_1[21], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[22], &sum_1[22], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[23], &sum_1[23], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[24], &sum_1[24], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[25], &sum_1[25], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[26], &sum_1[26], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[27], &sum_1[27], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[28], &sum_1[28], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[29], &sum_1[29], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[30], &sum_1[30], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[31], &sum_1[31], input_0, input_1, filter, output_depth_per_group);
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

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_2row32col_inplace(const float* input_data, 
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

  for (group = 0; group < groups - 1; group += 2) {
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

      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[0], &sum_1[0], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[1], &sum_1[1], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[2], &sum_1[2], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[3], &sum_1[3], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[4], &sum_1[4], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[5], &sum_1[5], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[6], &sum_1[6], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[7], &sum_1[7], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[8], &sum_1[8], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[9], &sum_1[9], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[10], &sum_1[10], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[11], &sum_1[11], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[12], &sum_1[12], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[13], &sum_1[13], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[14], &sum_1[14], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[15], &sum_1[15], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[16], &sum_1[16], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[17], &sum_1[17], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[18], &sum_1[18], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[19], &sum_1[19], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[20], &sum_1[20], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[21], &sum_1[21], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[22], &sum_1[22], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[23], &sum_1[23], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[24], &sum_1[24], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[25], &sum_1[25], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[26], &sum_1[26], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[27], &sum_1[27], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[28], &sum_1[28], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[29], &sum_1[29], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[30], &sum_1[30], input_0, input_1, filter, output_depth_per_group);
      filter++;
      group_mac_kernel5_2row_32col_fp_uniweight_IOHW(&sum_0[31], &sum_1[31], input_0, input_1, filter, output_depth_per_group);
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

tinyengine_status_fp group_conv_fp_kernel5_stride1_pad0_in5x5_out1x1_uniweight_2row32col_inplace_brutal(const float* input_data, 
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

  for(group = 0; group < groups - 1; group += 2) {
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

      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[0], &sum_1[0], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[1], &sum_1[1], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[2], &sum_1[2], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[3], &sum_1[3], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[4], &sum_1[4], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[5], &sum_1[5], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[6], &sum_1[6], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[7], &sum_1[7], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[8], &sum_1[8], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[9], &sum_1[9], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[10], &sum_1[10], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[11], &sum_1[11], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[12], &sum_1[12], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[13], &sum_1[13], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[14], &sum_1[14], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[15], &sum_1[15], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[16], &sum_1[16], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[17], &sum_1[17], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[18], &sum_1[18], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[19], &sum_1[19], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[20], &sum_1[20], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[21], &sum_1[21], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[22], &sum_1[22], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[23], &sum_1[23], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[24], &sum_1[24], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[25], &sum_1[25], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[26], &sum_1[26], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[27], &sum_1[27], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[28], &sum_1[28], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[29], &sum_1[29], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[30], &sum_1[30], input_0, input_1, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel5_2row_32col_fp_uniweight(&sum_0[31], &sum_1[31], input_0, input_1, filter);
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
