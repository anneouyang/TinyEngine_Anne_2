/* ----------------------------------------------------------------------
 * Name: group_conv_kernel4_stride1_pad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */
#include "tinyengine_function.h"
#include "nnfunctions.h"
#include "arm_nnfunctions.h"
#include "img2col_element.h"

#define DIM_KER_X (4U)
#define DIM_KER_Y (4U)

tinyengine_status group_conv_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_int8input_int8weight_inplace_revised(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const q31_t* bias_data, 
                 q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 q7_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;
  q31_t* tmp_output_buffer = norm_data;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 8) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    q7_t* two_column_buffer_0 = im2col_data; q7_t* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    q7_t* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; q7_t* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    q7_t* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; q7_t* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    q7_t* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; q7_t* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    const q7_t* src_0 = filter_data++; const q7_t* src_1 = filter_data++; const q7_t* src_2 = filter_data++; const q7_t* src_3 = filter_data++;
    const q7_t* src_4 = filter_data++; const q7_t* src_5 = filter_data++; const q7_t* src_6 = filter_data++; const q7_t* src_7 = filter_data++;

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
    q31_t* tmp_out_0 = tmp_output_buffer; q31_t* tmp_out_1 = &tmp_output_buffer[groups];
    q31_t* tmp_out_2 = &tmp_output_buffer[groups * 2]; q31_t* tmp_out_3 = &tmp_output_buffer[groups * 3];
    q31_t* tmp_out_4 = &tmp_output_buffer[groups * 4]; q31_t* tmp_out_5 = &tmp_output_buffer[groups * 5];
    q31_t* tmp_out_6 = &tmp_output_buffer[groups * 6]; q31_t* tmp_out_7 = &tmp_output_buffer[groups * 7];
    q31_t out_max_0 = 0; q31_t out_max_1 = 0;
    q31_t out_max_2 = 0; q31_t out_max_3 = 0;
    q31_t out_max_4 = 0; q31_t out_max_5 = 0;
    q31_t out_max_6 = 0; q31_t out_max_7 = 0;

    const q7_t* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
      const q7_t* src_8 = input++;
      const q7_t* src_9 = input++;
      const q7_t* src_10 = input++;
      const q7_t* src_11 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = *src_8;
          src_8 += input_depth;
          *two_column_buffer_1++ = *src_9;
          src_9 += input_depth;
          *two_column_buffer_2++ = *src_10;
          src_10 += input_depth;
          *two_column_buffer_3++ = *src_11;
          src_11 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const q7_t* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      const q7_t* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      const q7_t* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      const q7_t* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];

      const q7_t* filter = im2col_data;

      // We assume bias_data as zeros.
      q31_t sum_0[8] = {};
      q31_t sum_1[8] = {};
      q31_t sum_2[8] = {};
      q31_t sum_3[8] = {};
      
      /* Group Conv Computation */
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */ 
      assign_sum_to_group_tmp_output_buffer_4row8col_int8(tmp_out_0, tmp_out_1, tmp_out_2, tmp_out_3, tmp_out_4, tmp_out_5, tmp_out_6, tmp_out_7, 
                                                          sum_0, sum_1, sum_2, sum_3, 
                                                          &out_max_0, &out_max_1, &out_max_2, &out_max_3, &out_max_4, &out_max_5, &out_max_6, &out_max_7);
      tmp_out_0 += 4; tmp_out_1 += 4; tmp_out_2 += 4; tmp_out_3 += 4; tmp_out_4 += 4; tmp_out_5 += 4; tmp_out_6 += 4; tmp_out_7 += 4; 
    }

    /* Output Normalization */ 
    tmp_out_0 = tmp_output_buffer; tmp_out_1 = &tmp_output_buffer[groups];
    tmp_out_2 = &tmp_output_buffer[groups * 2]; tmp_out_3 = &tmp_output_buffer[groups * 3];
    tmp_out_4 = &tmp_output_buffer[groups * 4]; tmp_out_5 = &tmp_output_buffer[groups * 5];
    tmp_out_6 = &tmp_output_buffer[groups * 6]; tmp_out_7 = &tmp_output_buffer[groups * 7];

    q7_t* out_0 = &output_weight_data[i_output_depth * groups]; q7_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    q7_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; q7_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    q7_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; q7_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    q7_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; q7_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];

    int i_group;
    for (i_group = 0; i_group < groups; i_group++) {
      *out_0++ -= (q7_t) ((float)*tmp_out_0/(out_max_0/127) * scales[i_output_depth] * learning_rate);
      tmp_out_0++;
      *out_1++ -= (q7_t) ((float)*tmp_out_1/(out_max_1/127) * scales[i_output_depth + 1] * learning_rate);
      tmp_out_1++;
      *out_2++ -= (q7_t) ((float)*tmp_out_2/(out_max_2/127) * scales[i_output_depth + 2] * learning_rate);
      tmp_out_2++;
      *out_3++ -= (q7_t) ((float)*tmp_out_3/(out_max_3/127) * scales[i_output_depth + 3] * learning_rate);
      tmp_out_3++;
      *out_4++ -= (q7_t) ((float)*tmp_out_4/(out_max_4/127) * scales[i_output_depth + 4] * learning_rate);
      tmp_out_4++;
      *out_5++ -= (q7_t) ((float)*tmp_out_5/(out_max_5/127) * scales[i_output_depth + 5] * learning_rate);
      tmp_out_5++;
      *out_6++ -= (q7_t) ((float)*tmp_out_6/(out_max_6/127) * scales[i_output_depth + 6] * learning_rate);
      tmp_out_6++;
      *out_7++ -= (q7_t) ((float)*tmp_out_7/(out_max_7/127) * scales[i_output_depth + 7] * learning_rate);
      tmp_out_7++;
    }
  }

  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status group_conv_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_int8input_int8weight_inplace_revised(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const q31_t* bias_data, 
                 q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 q7_t* im2col_data, q31_t* norm_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;
  q31_t* tmp_output_buffer = norm_data;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 16) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    q7_t* two_column_buffer_0 = im2col_data; q7_t* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    q7_t* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; q7_t* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    q7_t* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; q7_t* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    q7_t* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; q7_t* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    q7_t* two_column_buffer_8 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8]; q7_t* two_column_buffer_9 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
    q7_t* two_column_buffer_10 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10]; q7_t* two_column_buffer_11 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
    q7_t* two_column_buffer_12 = &im2col_data[DIM_KER_X * DIM_KER_Y * 12]; q7_t* two_column_buffer_13 = &im2col_data[DIM_KER_X * DIM_KER_Y * 13];
    q7_t* two_column_buffer_14 = &im2col_data[DIM_KER_X * DIM_KER_Y * 14]; q7_t* two_column_buffer_15 = &im2col_data[DIM_KER_X * DIM_KER_Y * 15];
    const q7_t* src_0 = filter_data++; const q7_t* src_1 = filter_data++; const q7_t* src_2 = filter_data++; const q7_t* src_3 = filter_data++;
    const q7_t* src_4 = filter_data++; const q7_t* src_5 = filter_data++; const q7_t* src_6 = filter_data++; const q7_t* src_7 = filter_data++;
    const q7_t* src_8 = filter_data++; const q7_t* src_9 = filter_data++; const q7_t* src_10 = filter_data++; const q7_t* src_11 = filter_data++;
    const q7_t* src_12 = filter_data++; const q7_t* src_13 = filter_data++; const q7_t* src_14 = filter_data++; const q7_t* src_15 = filter_data++;

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
    q31_t* tmp_out_0 = tmp_output_buffer; q31_t* tmp_out_1 = &tmp_output_buffer[groups];
    q31_t* tmp_out_2 = &tmp_output_buffer[groups * 2]; q31_t* tmp_out_3 = &tmp_output_buffer[groups * 3];
    q31_t* tmp_out_4 = &tmp_output_buffer[groups * 4]; q31_t* tmp_out_5 = &tmp_output_buffer[groups * 5];
    q31_t* tmp_out_6 = &tmp_output_buffer[groups * 6]; q31_t* tmp_out_7 = &tmp_output_buffer[groups * 7];
    q31_t* tmp_out_8 = &tmp_output_buffer[groups * 8]; q31_t* tmp_out_9 = &tmp_output_buffer[groups * 9];
    q31_t* tmp_out_10 = &tmp_output_buffer[groups * 10]; q31_t* tmp_out_11 = &tmp_output_buffer[groups * 11];
    q31_t* tmp_out_12 = &tmp_output_buffer[groups * 12]; q31_t* tmp_out_13 = &tmp_output_buffer[groups * 13];
    q31_t* tmp_out_14 = &tmp_output_buffer[groups * 14]; q31_t* tmp_out_15 = &tmp_output_buffer[groups * 15];

    q31_t out_max_0 = 0; q31_t out_max_1 = 0; q31_t out_max_2 = 0; q31_t out_max_3 = 0; 
    q31_t out_max_4 = 0; q31_t out_max_5 = 0; q31_t out_max_6 = 0; q31_t out_max_7 = 0; 
    q31_t out_max_8 = 0; q31_t out_max_9 = 0; q31_t out_max_10 = 0; q31_t out_max_11 = 0; 
    q31_t out_max_12 = 0; q31_t out_max_13 = 0; q31_t out_max_14 = 0; q31_t out_max_15 = 0; 

    const q7_t* input = input_data;

    /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
    const q7_t* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
    const q7_t* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
    const q7_t* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
    const q7_t* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];

    // use variables
    q31_t in_q7x4;
    q31_t in_q15x2_1;
    q31_t in_q15x2_2;
    q31_t out_q15x2_1;
    q31_t out_q15x2_2;

    q15_t temp_buf[64] = {};
    q15_t* runtime_buf = (q15_t*) temp_buf;
    q7_q15_reordered_ele(input_0, runtime_buf);
    q7_q15_reordered_ele(input_0, runtime_buf);
    q7_q15_reordered_ele(input_0, runtime_buf);
    q7_q15_reordered_ele(input_0, runtime_buf);
    input_0 -= 16;
    q7_q15_reordered_ele(input_1, runtime_buf);
    q7_q15_reordered_ele(input_1, runtime_buf);
    q7_q15_reordered_ele(input_1, runtime_buf);
    q7_q15_reordered_ele(input_1, runtime_buf);
    input_1 -= 16;
    q7_q15_reordered_ele(input_2, runtime_buf);
    q7_q15_reordered_ele(input_2, runtime_buf);
    q7_q15_reordered_ele(input_2, runtime_buf);
    q7_q15_reordered_ele(input_2, runtime_buf);
    input_2 -= 16;
    q7_q15_reordered_ele(input_3, runtime_buf);
    q7_q15_reordered_ele(input_3, runtime_buf);
    q7_q15_reordered_ele(input_3, runtime_buf);
    q7_q15_reordered_ele(input_3, runtime_buf);
    input_3 -= 16;
    runtime_buf -= 64;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];
      const q7_t* src_16 = input++;
      const q7_t* src_17 = input++;
      const q7_t* src_18 = input++;
      const q7_t* src_19 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = *src_16;
          src_16 += input_depth;
          *two_column_buffer_1++ = *src_17;
          src_17 += input_depth;
          *two_column_buffer_2++ = *src_18;
          src_18 += input_depth;
          *two_column_buffer_3++ = *src_19;
          src_19 += input_depth;
        }
      }

      const q7_t* filter = im2col_data;

      // We assume bias_data as zeros.
      q31_t sum_0[16] = {};
      q31_t sum_1[16] = {};
      q31_t sum_2[16] = {};
      q31_t sum_3[16] = {};
      
      /* Group Conv Computation */
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;
      // group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
      // filter += DIM_KER_X * DIM_KER_Y;

      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input_simd(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], runtime_buf, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      assign_sum_to_group_tmp_output_buffer_4row16col_int8(tmp_out_0, tmp_out_1, tmp_out_2, tmp_out_3, tmp_out_4, tmp_out_5, tmp_out_6, tmp_out_7, 
                                                           tmp_out_8, tmp_out_9, tmp_out_10, tmp_out_11, tmp_out_12, tmp_out_13, tmp_out_14, tmp_out_15, 
                                                           sum_0, sum_1, sum_2, sum_3, 
                                                           &out_max_0, &out_max_1, &out_max_2, &out_max_3, &out_max_4, &out_max_5, &out_max_6, &out_max_7,
                                                           &out_max_8, &out_max_9, &out_max_10, &out_max_11, &out_max_12, &out_max_13, &out_max_14, &out_max_15);
      tmp_out_0 += 4; tmp_out_1 += 4; tmp_out_2 += 4; tmp_out_3 += 4; tmp_out_4 += 4; tmp_out_5 += 4; tmp_out_6 += 4; tmp_out_7 += 4; 
      tmp_out_8 += 4; tmp_out_9 += 4; tmp_out_10 += 4; tmp_out_11 += 4; tmp_out_12 += 4; tmp_out_13 += 4; tmp_out_14 += 4; tmp_out_15 += 4; 
    }

    /* Output Normalization */ 
    tmp_out_0 = tmp_output_buffer; tmp_out_1 = &tmp_output_buffer[groups];
    tmp_out_2 = &tmp_output_buffer[groups * 2]; tmp_out_3 = &tmp_output_buffer[groups * 3];
    tmp_out_4 = &tmp_output_buffer[groups * 4]; tmp_out_5 = &tmp_output_buffer[groups * 5];
    tmp_out_6 = &tmp_output_buffer[groups * 6]; tmp_out_7 = &tmp_output_buffer[groups * 7];
    tmp_out_8 = &tmp_output_buffer[groups * 8]; tmp_out_9 = &tmp_output_buffer[groups * 9];
    tmp_out_10 = &tmp_output_buffer[groups * 10]; tmp_out_11 = &tmp_output_buffer[groups * 11];
    tmp_out_12 = &tmp_output_buffer[groups * 12]; tmp_out_13 = &tmp_output_buffer[groups * 13];
    tmp_out_14 = &tmp_output_buffer[groups * 14]; tmp_out_15 = &tmp_output_buffer[groups * 15];

    q7_t* out_0 = &output_weight_data[i_output_depth * groups]; q7_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    q7_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; q7_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    q7_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; q7_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    q7_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; q7_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];
    q7_t* out_8 = &output_weight_data[(i_output_depth + 8) * groups]; q7_t* out_9 = &output_weight_data[(i_output_depth + 9) * groups]; 
    q7_t* out_10 = &output_weight_data[(i_output_depth + 10) * groups]; q7_t* out_11 = &output_weight_data[(i_output_depth + 11) * groups]; 
    q7_t* out_12 = &output_weight_data[(i_output_depth + 12) * groups]; q7_t* out_13 = &output_weight_data[(i_output_depth + 13) * groups]; 
    q7_t* out_14 = &output_weight_data[(i_output_depth + 14) * groups]; q7_t* out_15 = &output_weight_data[(i_output_depth + 15) * groups]; 

    int i_group;
    for (i_group = 0; i_group < groups; i_group++) {
      *out_0++ -= (q7_t) ((float)*tmp_out_0/(out_max_0/127) * scales[i_output_depth] * learning_rate);
      tmp_out_0++;
      *out_1++ -= (q7_t) ((float)*tmp_out_1/(out_max_1/127) * scales[i_output_depth + 1] * learning_rate);
      tmp_out_1++;
      *out_2++ -= (q7_t) ((float)*tmp_out_2/(out_max_2/127) * scales[i_output_depth + 2] * learning_rate);
      tmp_out_2++;
      *out_3++ -= (q7_t) ((float)*tmp_out_3/(out_max_3/127) * scales[i_output_depth + 3] * learning_rate);
      tmp_out_3++;
      *out_4++ -= (q7_t) ((float)*tmp_out_4/(out_max_4/127) * scales[i_output_depth + 4] * learning_rate);
      tmp_out_4++;
      *out_5++ -= (q7_t) ((float)*tmp_out_5/(out_max_5/127) * scales[i_output_depth + 5] * learning_rate);
      tmp_out_5++;
      *out_6++ -= (q7_t) ((float)*tmp_out_6/(out_max_6/127) * scales[i_output_depth + 6] * learning_rate);
      tmp_out_6++;
      *out_7++ -= (q7_t) ((float)*tmp_out_7/(out_max_7/127) * scales[i_output_depth + 7] * learning_rate);
      tmp_out_7++;
      *out_8++ -= (q7_t) ((float)*tmp_out_8/(out_max_8/127) * scales[i_output_depth + 8] * learning_rate);
      tmp_out_8++;
      *out_9++ -= (q7_t) ((float)*tmp_out_9/(out_max_9/127) * scales[i_output_depth + 9] * learning_rate);
      tmp_out_9++;
      *out_10++ -= (q7_t) ((float)*tmp_out_10/(out_max_10/127) * scales[i_output_depth + 10] * learning_rate);
      tmp_out_10++;
      *out_11++ -= (q7_t) ((float)*tmp_out_11/(out_max_11/127) * scales[i_output_depth + 11] * learning_rate);
      tmp_out_11++;
      *out_12++ -= (q7_t) ((float)*tmp_out_12/(out_max_12/127) * scales[i_output_depth + 12] * learning_rate);
      tmp_out_12++;
      *out_13++ -= (q7_t) ((float)*tmp_out_13/(out_max_13/127) * scales[i_output_depth + 13] * learning_rate);
      tmp_out_13++;
      *out_14++ -= (q7_t) ((float)*tmp_out_14/(out_max_14/127) * scales[i_output_depth + 14] * learning_rate);
      tmp_out_14++;
      *out_15++ -= (q7_t) ((float)*tmp_out_15/(out_max_15/127) * scales[i_output_depth + 15] * learning_rate);
      tmp_out_15++;
    }
  }

  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status group_conv_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row8col_int8input_int8weight_inplace_revised_noNORM(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const q31_t* bias_data, 
                 q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 q7_t* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 8) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    q7_t* two_column_buffer_0 = im2col_data; q7_t* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    q7_t* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; q7_t* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    q7_t* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; q7_t* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    q7_t* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; q7_t* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    const q7_t* src_0 = filter_data++; const q7_t* src_1 = filter_data++; const q7_t* src_2 = filter_data++; const q7_t* src_3 = filter_data++;
    const q7_t* src_4 = filter_data++; const q7_t* src_5 = filter_data++; const q7_t* src_6 = filter_data++; const q7_t* src_7 = filter_data++;

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
    q7_t* out_0 = &output_weight_data[i_output_depth * groups]; q7_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    q7_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; q7_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    q7_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; q7_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    q7_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; q7_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];

    const q7_t* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
      const q7_t* src_8 = input++;
      const q7_t* src_9 = input++;
      const q7_t* src_10 = input++;
      const q7_t* src_11 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = *src_8;
          src_8 += input_depth;
          *two_column_buffer_1++ = *src_9;
          src_9 += input_depth;
          *two_column_buffer_2++ = *src_10;
          src_10 += input_depth;
          *two_column_buffer_3++ = *src_11;
          src_11 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const q7_t* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8];
      const q7_t* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
      const q7_t* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10];
      const q7_t* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];

      const q7_t* filter = im2col_data;

      // We assume bias_data as zeros.
      q31_t sum_0[8] = {};
      q31_t sum_1[8] = {};
      q31_t sum_2[8] = {};
      q31_t sum_3[8] = {};
      
      /* Group Conv Computation */
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */  
      assign_sum_to_group_output_4row8col_int8(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, sum_0, sum_1, sum_2, sum_3, 
                                    output_activation_min, output_activation_max, scales, learning_rate, i_output_depth);
      out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4; out_4 += 4; out_5 += 4; out_6 += 4; out_7 += 4; 
    }
  }

  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status group_conv_kernel4_stride1_pad0_in4x4_out1x1_uniweight_4row16col_int8input_int8weight_inplace_revised_noNORM(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const q31_t* bias_data, 
                 q7_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 q7_t* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  int i_output_depth, i , j;
  int output_depth_per_group = output_depth / groups;

  for (i_output_depth = 0; i_output_depth < output_depth_per_group; i_output_depth += 16) {
    /* Alter the data format of filter_data from IHWO to OHWI and put it into im2col_data buffer */
    q7_t* two_column_buffer_0 = im2col_data; q7_t* two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y];
    q7_t* two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 2]; q7_t* two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 3];
    q7_t* two_column_buffer_4 = &im2col_data[DIM_KER_X * DIM_KER_Y * 4]; q7_t* two_column_buffer_5 = &im2col_data[DIM_KER_X * DIM_KER_Y * 5];
    q7_t* two_column_buffer_6 = &im2col_data[DIM_KER_X * DIM_KER_Y * 6]; q7_t* two_column_buffer_7 = &im2col_data[DIM_KER_X * DIM_KER_Y * 7];
    q7_t* two_column_buffer_8 = &im2col_data[DIM_KER_X * DIM_KER_Y * 8]; q7_t* two_column_buffer_9 = &im2col_data[DIM_KER_X * DIM_KER_Y * 9];
    q7_t* two_column_buffer_10 = &im2col_data[DIM_KER_X * DIM_KER_Y * 10]; q7_t* two_column_buffer_11 = &im2col_data[DIM_KER_X * DIM_KER_Y * 11];
    q7_t* two_column_buffer_12 = &im2col_data[DIM_KER_X * DIM_KER_Y * 12]; q7_t* two_column_buffer_13 = &im2col_data[DIM_KER_X * DIM_KER_Y * 13];
    q7_t* two_column_buffer_14 = &im2col_data[DIM_KER_X * DIM_KER_Y * 14]; q7_t* two_column_buffer_15 = &im2col_data[DIM_KER_X * DIM_KER_Y * 15];
    const q7_t* src_0 = filter_data++; const q7_t* src_1 = filter_data++; const q7_t* src_2 = filter_data++; const q7_t* src_3 = filter_data++;
    const q7_t* src_4 = filter_data++; const q7_t* src_5 = filter_data++; const q7_t* src_6 = filter_data++; const q7_t* src_7 = filter_data++;
    const q7_t* src_8 = filter_data++; const q7_t* src_9 = filter_data++; const q7_t* src_10 = filter_data++; const q7_t* src_11 = filter_data++;
    const q7_t* src_12 = filter_data++; const q7_t* src_13 = filter_data++; const q7_t* src_14 = filter_data++; const q7_t* src_15 = filter_data++;

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
    q7_t* out_0 = &output_weight_data[i_output_depth * groups]; q7_t* out_1 = &output_weight_data[(i_output_depth + 1) * groups];
    q7_t* out_2 = &output_weight_data[(i_output_depth + 2) * groups]; q7_t* out_3 = &output_weight_data[(i_output_depth + 3) * groups];
    q7_t* out_4 = &output_weight_data[(i_output_depth + 4) * groups]; q7_t* out_5 = &output_weight_data[(i_output_depth + 5) * groups];
    q7_t* out_6 = &output_weight_data[(i_output_depth + 6) * groups]; q7_t* out_7 = &output_weight_data[(i_output_depth + 7) * groups];
    q7_t* out_8 = &output_weight_data[(i_output_depth + 8) * groups]; q7_t* out_9 = &output_weight_data[(i_output_depth + 9) * groups];
    q7_t* out_10 = &output_weight_data[(i_output_depth + 10) * groups]; q7_t* out_11 = &output_weight_data[(i_output_depth + 11) * groups];
    q7_t* out_12 = &output_weight_data[(i_output_depth + 12) * groups]; q7_t* out_13 = &output_weight_data[(i_output_depth + 13) * groups];
    q7_t* out_14 = &output_weight_data[(i_output_depth + 14) * groups]; q7_t* out_15 = &output_weight_data[(i_output_depth + 15) * groups];

    const q7_t* input = input_data;

    /* Calculate 4 rows(input channels) at a time */
    uint16_t group_cnt = groups >> 2;
    while (group_cnt--) {
      /* Alter the data format of input_data from HWC to CHW and put it into im2col_data buffer */
      two_column_buffer_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      two_column_buffer_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      two_column_buffer_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      two_column_buffer_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];
      const q7_t* src_16 = input++;
      const q7_t* src_17 = input++;
      const q7_t* src_18 = input++;
      const q7_t* src_19 = input++;

      for (i = 0; i < input_height; i++) {
        for (j = 0; j < input_width; j++) {
          *two_column_buffer_0++ = *src_16;
          src_16 += input_depth;
          *two_column_buffer_1++ = *src_17;
          src_17 += input_depth;
          *two_column_buffer_2++ = *src_18;
          src_18 += input_depth;
          *two_column_buffer_3++ = *src_19;
          src_19 += input_depth;
        }
      }

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const q7_t* input_0 = &im2col_data[DIM_KER_X * DIM_KER_Y * 16];
      const q7_t* input_1 = &im2col_data[DIM_KER_X * DIM_KER_Y * 17];
      const q7_t* input_2 = &im2col_data[DIM_KER_X * DIM_KER_Y * 18];
      const q7_t* input_3 = &im2col_data[DIM_KER_X * DIM_KER_Y * 19];

      const q7_t* filter = im2col_data;

      // We assume bias_data as zeros.
      q31_t sum_0[16] = {};
      q31_t sum_1[16] = {};
      q31_t sum_2[16] = {};
      q31_t sum_3[16] = {};
      
      /* Group Conv Computation */
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[0], &sum_1[0], &sum_2[0], &sum_3[0], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[1], &sum_1[1], &sum_2[1], &sum_3[1], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[2], &sum_1[2], &sum_2[2], &sum_3[2], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[3], &sum_1[3], &sum_2[3], &sum_3[3], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[4], &sum_1[4], &sum_2[4], &sum_3[4], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[5], &sum_1[5], &sum_2[5], &sum_3[5], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[6], &sum_1[6], &sum_2[6], &sum_3[6], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[7], &sum_1[7], &sum_2[7], &sum_3[7], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[8], &sum_1[8], &sum_2[8], &sum_3[8], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[9], &sum_1[9], &sum_2[9], &sum_3[9], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[10], &sum_1[10], &sum_2[10], &sum_3[10], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[11], &sum_1[11], &sum_2[11], &sum_3[11], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[12], &sum_1[12], &sum_2[12], &sum_3[12], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[13], &sum_1[13], &sum_2[13], &sum_3[13], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[14], &sum_1[14], &sum_2[14], &sum_3[14], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;
      group_mac_kernel4_4row_uniweight_reuse_output_input(&sum_0[15], &sum_1[15], &sum_2[15], &sum_3[15], input_0, input_1, input_2, input_3, filter);
      filter += DIM_KER_X * DIM_KER_Y;

      /* Calculate outputs */      
      assign_sum_to_group_output_4row16col_int8(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, 
                                    sum_0, sum_1, sum_2, sum_3, output_activation_min, output_activation_max, scales, learning_rate, i_output_depth);
      out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4; out_4 += 4; out_5 += 4; out_6 += 4; out_7 += 4; 
      out_8 += 4; out_9 += 4; out_10 += 4; out_11 += 4; out_12 += 4; out_13 += 4; out_14 += 4; out_15 += 4; 
    }
  }

  /* Return to application */
  return STATE_SUCCESS;
}
