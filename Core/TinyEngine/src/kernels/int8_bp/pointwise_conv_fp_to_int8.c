/* ----------------------------------------------------------------------
 * Name: pointwise_conv_fp_to_int8.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"
#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8output_int8w(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const float* bias_data, 
                 q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const int32_t output_activation_min, const int32_t output_activation_max,
                 int16_t* im2col_data, float* norm_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  float* tmp_output_buffer = norm_data;
  float* tmp_output_buffer_start = tmp_output_buffer;
  q7_t* output = output_data;

  int i_element;
  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=10) {
      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      tmp_output_buffer = tmp_output_buffer_start;

      const int8_t* filter_0_int8 = &filter_data[i_ch_in * output_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_in + 1) * output_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_in + 2) * output_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_in + 3) * output_depth];
      const int8_t* filter_4_int8 = &filter_data[(i_ch_in + 4) * output_depth];
      const int8_t* filter_5_int8 = &filter_data[(i_ch_in + 5) * output_depth];
      const int8_t* filter_6_int8 = &filter_data[(i_ch_in + 6) * output_depth];
      const int8_t* filter_7_int8 = &filter_data[(i_ch_in + 7) * output_depth];
      const int8_t* filter_8_int8 = &filter_data[(i_ch_in + 8) * output_depth];
      const int8_t* filter_9_int8 = &filter_data[(i_ch_in + 9) * output_depth];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        float sum[8] = {};

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++;
        mac_1row_10col_fp_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);

        *tmp_output_buffer++ = sum[0];
        *tmp_output_buffer++ = sum[1];
        *tmp_output_buffer++ = sum[2];
        *tmp_output_buffer++ = sum[3];
        *tmp_output_buffer++ = sum[4];
        *tmp_output_buffer++ = sum[5];
        *tmp_output_buffer++ = sum[6];
        *tmp_output_buffer++ = sum[7];
      }

      /* Output Normalization */
      float* tmp_out_0 = tmp_output_buffer_start;
      float out_max_0 = 0;

      int i_output_depth;

      for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
        out_max_0 = MAX(out_max_0, fabsf(*tmp_out_0));
        tmp_out_0++;
      }

      tmp_out_0 = tmp_output_buffer_start;

      for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
        *output++ = (q7_t) (*tmp_out_0/(out_max_0/127));
        tmp_out_0++;
      }
    }
  }

  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status pointwise_conv_fp_4row4col_IOHW_int8output_int8w(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const float* bias_data, 
                 q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const int32_t output_activation_min, const int32_t output_activation_max,
                 int16_t* im2col_data, float* norm_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;
  float* tmp_output_buffer = norm_data;
  q7_t* output = output_data;

  for (i_element = 0; i_element/4 < num_elements/4; i_element+=4) {
    /* Initialize output data as 0 (assume bias == NULL) */
    int i;
    for(i = 0; i < output_depth * 4; i+=4) {
      tmp_output_buffer[i] = 0;
      tmp_output_buffer[i + 1] = 0;
      tmp_output_buffer[i + 2] = 0;
      tmp_output_buffer[i + 3] = 0;
    }

    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      float* out_0 = tmp_output_buffer;
      float* out_1 = &tmp_output_buffer[output_depth * 1];
      float* out_2 = &tmp_output_buffer[output_depth * 2];
      float* out_3 = &tmp_output_buffer[output_depth * 3];

      const float* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const float* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const float* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const float* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_data[i_ch_in * output_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_in + 1) * output_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_in + 2) * output_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_in + 3) * output_depth];
      float filter_0, filter_1, filter_2, filter_3;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[32] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[16], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[20], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[24], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_IOHW_forint8w(&sum[28], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_output_4row8col_noMINMAX(out_0, out_1, out_2, out_3, sum);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }
    }

    /* Output Normalization */
    float* tmp_out_0 = tmp_output_buffer;
    float* tmp_out_1 = &tmp_output_buffer[output_depth * 1];
    float* tmp_out_2 = &tmp_output_buffer[output_depth * 2];
    float* tmp_out_3 = &tmp_output_buffer[output_depth * 3];
    float out_max_0 = 0; float out_max_1 = 0; float out_max_2 = 0; float out_max_3 = 0; 
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, fabsf(*tmp_out_0));
      tmp_out_0++;

      out_max_1 = MAX(out_max_1, fabsf(*tmp_out_1));
      tmp_out_1++;

      out_max_2 = MAX(out_max_2, fabsf(*tmp_out_2));
      tmp_out_2++;

      out_max_3 = MAX(out_max_3, fabsf(*tmp_out_3));
      tmp_out_3++;
    }

    q7_t* out_0 = &output[i_element * output_depth];
    q7_t* out_1 = &output[(i_element + 1) * output_depth];
    q7_t* out_2 = &output[(i_element + 2) * output_depth];
    q7_t* out_3 = &output[(i_element + 3) * output_depth];
    tmp_out_0 = tmp_output_buffer;
    tmp_out_1 = &tmp_output_buffer[output_depth * 1];
    tmp_out_2 = &tmp_output_buffer[output_depth * 2];
    tmp_out_3 = &tmp_output_buffer[output_depth * 3];

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *out_0++ = (q7_t) (*tmp_out_0/(out_max_0/127));
      tmp_out_0++;

      *out_1++ = (q7_t) (*tmp_out_1/(out_max_1/127));
      tmp_out_1++;

      *out_2++ = (q7_t) (*tmp_out_2/(out_max_2/127));
      tmp_out_2++;

      *out_3++ = (q7_t) (*tmp_out_3/(out_max_3/127));
      tmp_out_3++;
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) {
    /* Initialize output data as 0 (assume bias == NULL) */
    int i;
    for(i = 0; i < output_depth; i++) {
      tmp_output_buffer[i] = 0;
    }

    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      float* out_0 = tmp_output_buffer;

      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_data[i_ch_in * output_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_in + 1) * output_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_in + 2) * output_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_in + 3) * output_depth];
      float filter_0, filter_1, filter_2, filter_3;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[8] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3);

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_output_1row8col_noMINMAX(out_0, sum);
        out_0 += 8;
      }
    }

    /* Output Normalization */
    float* tmp_out_0 = tmp_output_buffer;
    float out_max_0 = 0;
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, fabsf(*tmp_out_0));
      tmp_out_0++;
    }

    q7_t* out_0 = &output[(num_elements - leftover_elements) * output_depth];
    tmp_out_0 = tmp_output_buffer;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *out_0++ = (q7_t) (*tmp_out_0/(out_max_0/127));
      tmp_out_0++;
    }

    leftover_elements--;
  }
  
  /* Return to application */
  return STATE_SUCCESS;
}
