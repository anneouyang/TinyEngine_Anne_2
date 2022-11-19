/* ----------------------------------------------------------------------
 * Name: group_pointwise_conv_fp.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight_int8input_inplace(const int8_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  (void) input_height;
  (void) input_width;

  int group;
  int output_depth_per_group = output_depth / groups;

  for (group = 0; group < groups; group++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth_per_group; i_ch_out+=10) {
      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float input_0 = (float)input_data[group];
      const float filter[10] = {filter_data[i_ch_out], filter_data[i_ch_out + 1], filter_data[i_ch_out + 2], filter_data[i_ch_out + 3], filter_data[i_ch_out + 4], 
                      filter_data[i_ch_out + 5], filter_data[i_ch_out + 6], filter_data[i_ch_out + 7], filter_data[i_ch_out + 8], filter_data[i_ch_out + 9]};

      uint16_t col_count_div10 = (output_depth_per_group * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        // Assume bias_data as NULL
        float sum[10] = {};

        sum[0] += input_0 * filter[0];
        sum[1] += input_0 * filter[1];
        sum[2] += input_0 * filter[2];
        sum[3] += input_0 * filter[3];
        sum[4] += input_0 * filter[4];
        sum[5] += input_0 * filter[5];
        sum[6] += input_0 * filter[6];
        sum[7] += input_0 * filter[7];
        sum[8] += input_0 * filter[8];
        sum[9] += input_0 * filter[9];

        /*
        *out++ -= round(MIN(MAX(sum[0], output_activation_min), output_activation_max) * scales[i_ch_out] * learning_rate);
        *out++ -= round(MIN(MAX(sum[1], output_activation_min), output_activation_max) * scales[i_ch_out + 1] * learning_rate);
        *out++ -= round(MIN(MAX(sum[2], output_activation_min), output_activation_max) * scales[i_ch_out + 2] * learning_rate);
        *out++ -= round(MIN(MAX(sum[3], output_activation_min), output_activation_max) * scales[i_ch_out + 3] * learning_rate);
        *out++ -= round(MIN(MAX(sum[4], output_activation_min), output_activation_max) * scales[i_ch_out + 4] * learning_rate);
        *out++ -= round(MIN(MAX(sum[5], output_activation_min), output_activation_max) * scales[i_ch_out + 5] * learning_rate);
        *out++ -= round(MIN(MAX(sum[6], output_activation_min), output_activation_max) * scales[i_ch_out + 6] * learning_rate);
        *out++ -= round(MIN(MAX(sum[7], output_activation_min), output_activation_max) * scales[i_ch_out + 7] * learning_rate);
        *out++ -= round(MIN(MAX(sum[8], output_activation_min), output_activation_max) * scales[i_ch_out + 8] * learning_rate);
        *out -= round(MIN(MAX(sum[9], output_activation_min), output_activation_max) * scales[i_ch_out + 9] * learning_rate);
        */
        output_weight_data[i_ch_out + group] -= MIN(MAX(sum[0], output_activation_min), output_activation_max) * scales[i_ch_out] * learning_rate;
        output_weight_data[(i_ch_out + 1) * groups + group] -= MIN(MAX(sum[1], output_activation_min), output_activation_max) * scales[i_ch_out + 1] * learning_rate;
        output_weight_data[(i_ch_out + 2) * groups + group] -= MIN(MAX(sum[2], output_activation_min), output_activation_max) * scales[i_ch_out + 2] * learning_rate;
        output_weight_data[(i_ch_out + 3) * groups + group] -= MIN(MAX(sum[3], output_activation_min), output_activation_max) * scales[i_ch_out + 3] * learning_rate;
        output_weight_data[(i_ch_out + 4) * groups + group] -= MIN(MAX(sum[4], output_activation_min), output_activation_max) * scales[i_ch_out + 4] * learning_rate;
        output_weight_data[(i_ch_out + 5) * groups + group] -= MIN(MAX(sum[5], output_activation_min), output_activation_max) * scales[i_ch_out + 5] * learning_rate;
        output_weight_data[(i_ch_out + 6) * groups + group] -= MIN(MAX(sum[6], output_activation_min), output_activation_max) * scales[i_ch_out + 6] * learning_rate;
        output_weight_data[(i_ch_out + 7) * groups + group] -= MIN(MAX(sum[7], output_activation_min), output_activation_max) * scales[i_ch_out + 7] * learning_rate;
        output_weight_data[(i_ch_out + 8) * groups + group] -= MIN(MAX(sum[8], output_activation_min), output_activation_max) * scales[i_ch_out + 8] * learning_rate;
        output_weight_data[(i_ch_out + 9) * groups + group] -= MIN(MAX(sum[9], output_activation_min), output_activation_max) * scales[i_ch_out + 9] * learning_rate;
      }
    }
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight_inplace(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  (void) input_height;
  (void) input_width;

  int group;
  int output_depth_per_group = output_depth / groups;

  for(group = 0; group < groups; group++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth_per_group; i_ch_out+=10) {
      //int8_t* out = &output_weight_data[output_depth_per_group * group + i_ch_out];
      /*float sum[10] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], 
                   bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7], bias_data[i_ch_out + 8], bias_data[i_ch_out + 9]};*/

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float input_0 = input_data[group];
      const float filter[10] = {filter_data[i_ch_out], filter_data[i_ch_out + 1], filter_data[i_ch_out + 2], filter_data[i_ch_out + 3], filter_data[i_ch_out + 4], 
                      filter_data[i_ch_out + 5], filter_data[i_ch_out + 6], filter_data[i_ch_out + 7], filter_data[i_ch_out + 8], filter_data[i_ch_out + 9]};

      uint16_t col_count_div10 = (output_depth_per_group * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        // Assume bias_data as NULL
        float sum[10] = {};

        sum[0] += input_0 * filter[0];
        sum[1] += input_0 * filter[1];
        sum[2] += input_0 * filter[2];
        sum[3] += input_0 * filter[3];
        sum[4] += input_0 * filter[4];
        sum[5] += input_0 * filter[5];
        sum[6] += input_0 * filter[6];
        sum[7] += input_0 * filter[7];
        sum[8] += input_0 * filter[8];
        sum[9] += input_0 * filter[9];

        /*
        *out++ -= round(MIN(MAX(sum[0], output_activation_min), output_activation_max) * scales[i_ch_out] * learning_rate);
        *out++ -= round(MIN(MAX(sum[1], output_activation_min), output_activation_max) * scales[i_ch_out + 1] * learning_rate);
        *out++ -= round(MIN(MAX(sum[2], output_activation_min), output_activation_max) * scales[i_ch_out + 2] * learning_rate);
        *out++ -= round(MIN(MAX(sum[3], output_activation_min), output_activation_max) * scales[i_ch_out + 3] * learning_rate);
        *out++ -= round(MIN(MAX(sum[4], output_activation_min), output_activation_max) * scales[i_ch_out + 4] * learning_rate);
        *out++ -= round(MIN(MAX(sum[5], output_activation_min), output_activation_max) * scales[i_ch_out + 5] * learning_rate);
        *out++ -= round(MIN(MAX(sum[6], output_activation_min), output_activation_max) * scales[i_ch_out + 6] * learning_rate);
        *out++ -= round(MIN(MAX(sum[7], output_activation_min), output_activation_max) * scales[i_ch_out + 7] * learning_rate);
        *out++ -= round(MIN(MAX(sum[8], output_activation_min), output_activation_max) * scales[i_ch_out + 8] * learning_rate);
        *out -= round(MIN(MAX(sum[9], output_activation_min), output_activation_max) * scales[i_ch_out + 9] * learning_rate);
        */
        output_weight_data[i_ch_out + group] -= MIN(MAX(sum[0], output_activation_min), output_activation_max) * scales[i_ch_out] * learning_rate;
        output_weight_data[(i_ch_out + 1) * groups + group] -= MIN(MAX(sum[1], output_activation_min), output_activation_max) * scales[i_ch_out + 1] * learning_rate;
        output_weight_data[(i_ch_out + 2) * groups + group] -= MIN(MAX(sum[2], output_activation_min), output_activation_max) * scales[i_ch_out + 2] * learning_rate;
        output_weight_data[(i_ch_out + 3) * groups + group] -= MIN(MAX(sum[3], output_activation_min), output_activation_max) * scales[i_ch_out + 3] * learning_rate;
        output_weight_data[(i_ch_out + 4) * groups + group] -= MIN(MAX(sum[4], output_activation_min), output_activation_max) * scales[i_ch_out + 4] * learning_rate;
        output_weight_data[(i_ch_out + 5) * groups + group] -= MIN(MAX(sum[5], output_activation_min), output_activation_max) * scales[i_ch_out + 5] * learning_rate;
        output_weight_data[(i_ch_out + 6) * groups + group] -= MIN(MAX(sum[6], output_activation_min), output_activation_max) * scales[i_ch_out + 6] * learning_rate;
        output_weight_data[(i_ch_out + 7) * groups + group] -= MIN(MAX(sum[7], output_activation_min), output_activation_max) * scales[i_ch_out + 7] * learning_rate;
        output_weight_data[(i_ch_out + 8) * groups + group] -= MIN(MAX(sum[8], output_activation_min), output_activation_max) * scales[i_ch_out + 8] * learning_rate;
        output_weight_data[(i_ch_out + 9) * groups + group] -= MIN(MAX(sum[9], output_activation_min), output_activation_max) * scales[i_ch_out + 9] * learning_rate;
      }
    }
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col_uniweight(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups) {
  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int group;
  //int input_depth_per_group = input_depth / groups;
  int output_depth_per_group = output_depth / groups;

  //const int num_elements = output_height * output_width;

  for(group = 0; group < groups; group++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth_per_group; i_ch_out+=10) {
      //int cur_channel = output_depth_per_group * group + i_ch_out;

      float* out = &output_data[output_depth_per_group * group + i_ch_out];
      float sum[10] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], 
                   bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7], bias_data[i_ch_out + 8], bias_data[i_ch_out + 9]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float input_0 = input_data[group];
      float filter[10] = {filter_data[i_ch_out], filter_data[i_ch_out + 1], filter_data[i_ch_out + 2], filter_data[i_ch_out + 3], filter_data[i_ch_out + 4], 
                      filter_data[i_ch_out + 5], filter_data[i_ch_out + 6], filter_data[i_ch_out + 7], filter_data[i_ch_out + 8], filter_data[i_ch_out + 9]};

      uint16_t col_count_div10 = (output_depth_per_group * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        sum[0] += input_0 * filter[0];
        sum[1] += input_0 * filter[1];
        sum[2] += input_0 * filter[2];
        sum[3] += input_0 * filter[3];
        sum[4] += input_0 * filter[4];
        sum[5] += input_0 * filter[5];
        sum[6] += input_0 * filter[6];
        sum[7] += input_0 * filter[7];
        sum[8] += input_0 * filter[8];
        sum[9] += input_0 * filter[9];
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[8], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[9], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp group_pointwise_conv_fp_in1x1_out1x1_1row10col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups) {
  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int group;
  int input_depth_per_group = input_depth / groups;
  int output_depth_per_group = output_depth / groups;

  //const int num_elements = output_height * output_width;

  for(group = 0; group < groups; group++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth_per_group; i_ch_out+=10) {
      int cur_channel = output_depth_per_group * group + i_ch_out;

      float* out = &output_data[cur_channel];
      float sum[10] = {bias_data[cur_channel], bias_data[cur_channel + 1], bias_data[cur_channel + 2], bias_data[cur_channel + 3], bias_data[cur_channel + 4], 
                   bias_data[cur_channel + 5], bias_data[cur_channel + 6], bias_data[cur_channel + 7], bias_data[cur_channel + 8], bias_data[cur_channel + 9]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float input_0 = input_data[input_depth_per_group * group];
      float filter[10] = {filter_data[cur_channel], filter_data[cur_channel + 1], filter_data[cur_channel + 2], filter_data[cur_channel + 3], filter_data[cur_channel + 4], 
                      filter_data[cur_channel + 5], filter_data[cur_channel + 6], filter_data[cur_channel + 7], filter_data[cur_channel + 8], filter_data[cur_channel + 9]};

      uint16_t col_count_div10 = (output_depth_per_group * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        sum[0] += input_0 * filter[0];
        sum[1] += input_0 * filter[1];
        sum[2] += input_0 * filter[2];
        sum[3] += input_0 * filter[3];
        sum[4] += input_0 * filter[4];
        sum[5] += input_0 * filter[5];
        sum[6] += input_0 * filter[6];
        sum[7] += input_0 * filter[7];
        sum[8] += input_0 * filter[8];
        sum[9] += input_0 * filter[9];
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[8], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[9], output_activation_min), output_activation_max);
    }
  }
}
