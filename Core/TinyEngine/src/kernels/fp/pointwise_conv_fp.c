/* ----------------------------------------------------------------------
 * Name: pointwise_conv_fp.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status_fp pointwise_conv_fp_1row16col_10inputdepth(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=16) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[16] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7],
                      bias_data[i_ch_out + 8], bias_data[i_ch_out + 9], bias_data[i_ch_out + 10], bias_data[i_ch_out + 11], bias_data[i_ch_out + 12], bias_data[i_ch_out + 13], bias_data[i_ch_out + 14], bias_data[i_ch_out + 15]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];
      const float* filter_8 = &filter_data[(i_ch_out + 8) * input_depth];
      const float* filter_9 = &filter_data[(i_ch_out + 9) * input_depth];
      const float* filter_10 = &filter_data[(i_ch_out + 10) * input_depth];
      const float* filter_11 = &filter_data[(i_ch_out + 11) * input_depth];
      const float* filter_12 = &filter_data[(i_ch_out + 12) * input_depth];
      const float* filter_13 = &filter_data[(i_ch_out + 13) * input_depth];
      const float* filter_14 = &filter_data[(i_ch_out + 14) * input_depth];
      const float* filter_15 = &filter_data[(i_ch_out + 15) * input_depth];

      uint16_t col_count_div10 = (input_depth * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 
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
      *out++ = MIN(MAX(sum[9], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[10], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[11], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[12], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[13], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[14], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[15], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_1row16col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=16) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[16] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7],
                      bias_data[i_ch_out + 8], bias_data[i_ch_out + 9], bias_data[i_ch_out + 10], bias_data[i_ch_out + 11], bias_data[i_ch_out + 12], bias_data[i_ch_out + 13], bias_data[i_ch_out + 14], bias_data[i_ch_out + 15]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];
      const float* filter_8 = &filter_data[(i_ch_out + 8) * input_depth];
      const float* filter_9 = &filter_data[(i_ch_out + 9) * input_depth];
      const float* filter_10 = &filter_data[(i_ch_out + 10) * input_depth];
      const float* filter_11 = &filter_data[(i_ch_out + 11) * input_depth];
      const float* filter_12 = &filter_data[(i_ch_out + 12) * input_depth];
      const float* filter_13 = &filter_data[(i_ch_out + 13) * input_depth];
      const float* filter_14 = &filter_data[(i_ch_out + 14) * input_depth];
      const float* filter_15 = &filter_data[(i_ch_out + 15) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 

        mac_1row_16col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
        filter_8++; filter_9++; filter_10++; filter_11++; filter_12++; filter_13++; filter_14++; filter_15++; 
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
      *out++ = MIN(MAX(sum[9], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[10], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[11], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[12], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[13], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[14], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[15], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_1row4col_10inputdepth(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=4) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      //float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};
      float sum[4] = {};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];

      uint16_t col_count_div10 = (input_depth * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++;
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[3], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_4row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[32] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      const float* input_1 = &input_data[(i_element + 1) * input_depth];
      const float* input_2 = &input_data[(i_element + 2) * input_depth];
      const float* input_3 = &input_data[(i_element + 3) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_4row_8col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_4row_8col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_4row_8col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_4row_8col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      out += output_depth - 7;
      *out++ = MIN(MAX(sum[8], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[9], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[10], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[11], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[12], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[13], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[14], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[15], output_activation_min), output_activation_max);
      out += output_depth - 7;
      *out++ = MIN(MAX(sum[16], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[17], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[18], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[19], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[20], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[21], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[22], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[23], output_activation_min), output_activation_max);
      out += output_depth - 7;
      *out++ = MIN(MAX(sum[24], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[25], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[26], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[27], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[28], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[29], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[30], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[31], output_activation_min), output_activation_max);
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[(num_elements - leftover_elements) * output_depth + i_ch_out];
      float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                      bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_2row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element+=2) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[16] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      const float* input_1 = &input_data[(i_element + 1) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_2row_8col_fp(sum, input_0, input_1, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_2row_8col_fp(sum, input_0, input_1, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_2row_8col_fp(sum, input_0, input_1, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_2row_8col_fp(sum, input_0, input_1, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; input_1++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      out += output_depth - 7;
      *out++ = MIN(MAX(sum[8], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[9], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[10], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[11], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[12], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[13], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[14], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[15], output_activation_min), output_activation_max);
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x1;

  while (leftover_elements) { 
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[(num_elements - leftover_elements) * output_depth + i_ch_out];
      float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                      bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++;
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8w_partialCH(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  float* out = output_data;

  int i_element;
  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=10) {
      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const int8_t* filter_0_int8 = &filter_sram[i_ch_in * first_k_channel];
      const int8_t* filter_1_int8 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const int8_t* filter_2_int8 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const int8_t* filter_3_int8 = &filter_sram[(i_ch_in + 3) * first_k_channel];
      const int8_t* filter_4_int8 = &filter_sram[(i_ch_in + 4) * first_k_channel];
      const int8_t* filter_5_int8 = &filter_sram[(i_ch_in + 5) * first_k_channel];
      const int8_t* filter_6_int8 = &filter_sram[(i_ch_in + 6) * first_k_channel];
      const int8_t* filter_7_int8 = &filter_sram[(i_ch_in + 7) * first_k_channel];
      const int8_t* filter_8_int8 = &filter_sram[(i_ch_in + 8) * first_k_channel];
      const int8_t* filter_9_int8 = &filter_sram[(i_ch_in + 9) * first_k_channel];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9;

      /* Compute weights in SRAM */
      uint16_t col_count_div8 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 3;
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

        *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }

      filter_0_int8 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1_int8 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2_int8 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3_int8 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];
      filter_4_int8 = &filter_flash[(i_ch_in + 4) * (output_depth - first_k_channel)];
      filter_5_int8 = &filter_flash[(i_ch_in + 5) * (output_depth - first_k_channel)];
      filter_6_int8 = &filter_flash[(i_ch_in + 6) * (output_depth - first_k_channel)];
      filter_7_int8 = &filter_flash[(i_ch_in + 7) * (output_depth - first_k_channel)];
      filter_8_int8 = &filter_flash[(i_ch_in + 8) * (output_depth - first_k_channel)];
      filter_9_int8 = &filter_flash[(i_ch_in + 9) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div8 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 3;
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

        *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_1row10col_10inputdepth_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  float* out = output_data;

  int i_element;
  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=10) {
      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

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

        *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out++ = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_1row8col_10inputdepth_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      /*float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], 
      bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};*/
      float sum[8] = {};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const int8_t* filter_0_int8 = &filter_data[i_ch_out * input_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_out + 1) * input_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_out + 2) * input_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_out + 3) * input_depth];
      const int8_t* filter_4_int8 = &filter_data[(i_ch_out + 4) * input_depth];
      const int8_t* filter_5_int8 = &filter_data[(i_ch_out + 5) * input_depth];
      const int8_t* filter_6_int8 = &filter_data[(i_ch_out + 6) * input_depth];
      const int8_t* filter_7_int8 = &filter_data[(i_ch_out + 7) * input_depth];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7;

      uint16_t col_count_div10 = (input_depth * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_1row8col_10inputdepth(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      //float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};
      float sum[8] = {};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];

      uint16_t col_count_div10 = (input_depth * DIM_KER_X * DIM_KER_Y) / 10;

      while (col_count_div10--) {
        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_1row8col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=8) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3], bias_data[i_ch_out + 4], bias_data[i_ch_out + 5], bias_data[i_ch_out + 6], bias_data[i_ch_out + 7]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];
      const float* filter_4 = &filter_data[(i_ch_out + 4) * input_depth];
      const float* filter_5 = &filter_data[(i_ch_out + 5) * input_depth];
      const float* filter_6 = &filter_data[(i_ch_out + 6) * input_depth];
      const float* filter_7 = &filter_data[(i_ch_out + 7) * input_depth];

      uint16_t col_count_div8 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 

        mac_1row_8col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
    }
  }
}

tinyengine_status_fp pointwise_conv_fp_4row16col_IOHW_int8w_partialCH(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;

  /* Initialize output data as 0 (assume bias == NULL) */
  for(i_element = 0; i_element < output_depth*num_elements; i_element++) {
    output_data[i_element] = 0;
  }

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=16) {
      float* out_0 = &output_data[i_element * output_depth];
      float* out_1 = &output_data[(i_element + 1) * output_depth];
      float* out_2 = &output_data[(i_element + 2) * output_depth];
      float* out_3 = &output_data[(i_element + 3) * output_depth];

      const float* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const float* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const float* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const float* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_sram[i_ch_in * first_k_channel];
      const int8_t* filter_1_int8 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const int8_t* filter_2_int8 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const int8_t* filter_3_int8 = &filter_sram[(i_ch_in + 3) * first_k_channel];
      const int8_t* filter_4_int8 = &filter_sram[(i_ch_in + 4) * first_k_channel];
      const int8_t* filter_5_int8 = &filter_sram[(i_ch_in + 5) * first_k_channel];
      const int8_t* filter_6_int8 = &filter_sram[(i_ch_in + 6) * first_k_channel];
      const int8_t* filter_7_int8 = &filter_sram[(i_ch_in + 7) * first_k_channel];
      const int8_t* filter_8_int8 = &filter_sram[(i_ch_in + 8) * first_k_channel];
      const int8_t* filter_9_int8 = &filter_sram[(i_ch_in + 9) * first_k_channel];
      const int8_t* filter_10_int8 = &filter_sram[(i_ch_in + 10) * first_k_channel];
      const int8_t* filter_11_int8 = &filter_sram[(i_ch_in + 11) * first_k_channel];
      const int8_t* filter_12_int8 = &filter_sram[(i_ch_in + 12) * first_k_channel];
      const int8_t* filter_13_int8 = &filter_sram[(i_ch_in + 13) * first_k_channel];
      const int8_t* filter_14_int8 = &filter_sram[(i_ch_in + 14) * first_k_channel];
      const int8_t* filter_15_int8 = &filter_sram[(i_ch_in + 15) * first_k_channel];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15;

      /* Compute weights in SRAM */
      uint16_t col_count_div4 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 2;
      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[16] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[8], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[9], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[10], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[11], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[12], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[13], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[14], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[15], output_activation_min), output_activation_max);
      }

      filter_0_int8 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1_int8 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2_int8 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3_int8 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];
      filter_4_int8 = &filter_flash[(i_ch_in + 4) * (output_depth - first_k_channel)];
      filter_5_int8 = &filter_flash[(i_ch_in + 5) * (output_depth - first_k_channel)];
      filter_6_int8 = &filter_flash[(i_ch_in + 6) * (output_depth - first_k_channel)];
      filter_7_int8 = &filter_flash[(i_ch_in + 7) * (output_depth - first_k_channel)];
      filter_8_int8 = &filter_flash[(i_ch_in + 8) * (output_depth - first_k_channel)];
      filter_9_int8 = &filter_flash[(i_ch_in + 9) * (output_depth - first_k_channel)];
      filter_10_int8 = &filter_flash[(i_ch_in + 10) * (output_depth - first_k_channel)];
      filter_11_int8 = &filter_flash[(i_ch_in + 11) * (output_depth - first_k_channel)];
      filter_12_int8 = &filter_flash[(i_ch_in + 12) * (output_depth - first_k_channel)];
      filter_13_int8 = &filter_flash[(i_ch_in + 13) * (output_depth - first_k_channel)];
      filter_14_int8 = &filter_flash[(i_ch_in + 14) * (output_depth - first_k_channel)];
      filter_15_int8 = &filter_flash[(i_ch_in + 15) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div4 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 2;
      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[16] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[8], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[9], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[10], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[11], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[12], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[13], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[14], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[15], output_activation_min), output_activation_max);
      }
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=8) {
      float* out_0 = &output_data[(num_elements - leftover_elements) * output_depth];

      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_sram[i_ch_in * first_k_channel];
      const int8_t* filter_1_int8 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const int8_t* filter_2_int8 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const int8_t* filter_3_int8 = &filter_sram[(i_ch_in + 3) * first_k_channel];
      const int8_t* filter_4_int8 = &filter_sram[(i_ch_in + 4) * first_k_channel];
      const int8_t* filter_5_int8 = &filter_sram[(i_ch_in + 5) * first_k_channel];
      const int8_t* filter_6_int8 = &filter_sram[(i_ch_in + 6) * first_k_channel];
      const int8_t* filter_7_int8 = &filter_sram[(i_ch_in + 7) * first_k_channel];
      const int8_t* filter_8_int8 = &filter_sram[(i_ch_in + 8) * first_k_channel];
      const int8_t* filter_9_int8 = &filter_sram[(i_ch_in + 9) * first_k_channel];
      const int8_t* filter_10_int8 = &filter_sram[(i_ch_in + 10) * first_k_channel];
      const int8_t* filter_11_int8 = &filter_sram[(i_ch_in + 11) * first_k_channel];
      const int8_t* filter_12_int8 = &filter_sram[(i_ch_in + 12) * first_k_channel];
      const int8_t* filter_13_int8 = &filter_sram[(i_ch_in + 13) * first_k_channel];
      const int8_t* filter_14_int8 = &filter_sram[(i_ch_in + 14) * first_k_channel];
      const int8_t* filter_15_int8 = &filter_sram[(i_ch_in + 15) * first_k_channel];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15;

      /* Compute weights in SRAM */
      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;
      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[8] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }

      filter_0_int8 = filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1_int8 = filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2_int8 = filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3_int8 = filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];
      filter_4_int8 = filter_flash[(i_ch_in + 4) * (output_depth - first_k_channel)];
      filter_5_int8 = filter_flash[(i_ch_in + 5) * (output_depth - first_k_channel)];
      filter_6_int8 = filter_flash[(i_ch_in + 6) * (output_depth - first_k_channel)];
      filter_7_int8 = filter_flash[(i_ch_in + 7) * (output_depth - first_k_channel)];
      filter_8_int8 = filter_flash[(i_ch_in + 8) * (output_depth - first_k_channel)];
      filter_9_int8 = filter_flash[(i_ch_in + 9) * (output_depth - first_k_channel)];
      filter_10_int8 = filter_flash[(i_ch_in + 10) * (output_depth - first_k_channel)];
      filter_11_int8 = filter_flash[(i_ch_in + 11) * (output_depth - first_k_channel)];
      filter_12_int8 = filter_flash[(i_ch_in + 12) * (output_depth - first_k_channel)];
      filter_13_int8 = filter_flash[(i_ch_in + 13) * (output_depth - first_k_channel)];
      filter_14_int8 = filter_flash[(i_ch_in + 14) * (output_depth - first_k_channel)];
      filter_15_int8 = filter_flash[(i_ch_in + 15) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div8 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 3;
      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[8] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_4row16col_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;

  /* Initialize output data as 0 (assume bias == NULL) */
  for(i_element = 0; i_element < output_depth*num_elements; i_element++) {
    output_data[i_element] = 0;
  }

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=16) {
      float* out_0 = &output_data[i_element * output_depth];
      float* out_1 = &output_data[(i_element + 1) * output_depth];
      float* out_2 = &output_data[(i_element + 2) * output_depth];
      float* out_3 = &output_data[(i_element + 3) * output_depth];

      const float* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const float* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const float* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const float* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

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
      const int8_t* filter_10_int8 = &filter_data[(i_ch_in + 10) * output_depth];
      const int8_t* filter_11_int8 = &filter_data[(i_ch_in + 11) * output_depth];
      const int8_t* filter_12_int8 = &filter_data[(i_ch_in + 12) * output_depth];
      const int8_t* filter_13_int8 = &filter_data[(i_ch_in + 13) * output_depth];
      const int8_t* filter_14_int8 = &filter_data[(i_ch_in + 14) * output_depth];
      const int8_t* filter_15_int8 = &filter_data[(i_ch_in + 15) * output_depth];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15;

      uint16_t col_count_div4 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[16] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        /*
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[16], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[20], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[24], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_4row_16col_fp_IOHW_forint8w(&sum[28], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);
        */

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[8], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[9], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[10], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[11], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[12], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[13], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[14], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[15], output_activation_min), output_activation_max);
        /*
        *out_0++ += MIN(MAX(sum[16], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[17], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[18], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[19], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[20], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[21], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[22], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[23], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[24], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[25], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[26], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[27], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[28], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[29], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[30], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[31], output_activation_min), output_activation_max);
        */
      }
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=8) {
      float* out_0 = &output_data[(num_elements - leftover_elements) * output_depth];

      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

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
      const int8_t* filter_10_int8 = &filter_data[(i_ch_in + 10) * output_depth];
      const int8_t* filter_11_int8 = &filter_data[(i_ch_in + 11) * output_depth];
      const int8_t* filter_12_int8 = &filter_data[(i_ch_in + 12) * output_depth];
      const int8_t* filter_13_int8 = &filter_data[(i_ch_in + 13) * output_depth];
      const int8_t* filter_14_int8 = &filter_data[(i_ch_in + 14) * output_depth];
      const int8_t* filter_15_int8 = &filter_data[(i_ch_in + 15) * output_depth];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15;


      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[8] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        filter_8 = (float)*filter_8_int8++; filter_9 = (float)*filter_9_int8++; filter_10 = (float)*filter_10_int8++; filter_11 = (float)*filter_11_int8++;
        filter_12 = (float)*filter_12_int8++; filter_13 = (float)*filter_13_int8++; filter_14 = (float)*filter_14_int8++; filter_15 = (float)*filter_15_int8++;
        mac_1row_16col_fp_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7,
                                        filter_8, filter_9, filter_10, filter_11, filter_12, filter_13, filter_14, filter_15);

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_4row8col_IOHW_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;

  /* Initialize output data as 0 (assume bias == NULL) */
  for(i_element = 0; i_element < output_depth*num_elements; i_element++) {
    output_data[i_element] = 0;
  }

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=8) {
      float* out_0 = &output_data[i_element * output_depth];
      float* out_1 = &output_data[(i_element + 1) * output_depth];
      float* out_2 = &output_data[(i_element + 2) * output_depth];
      float* out_3 = &output_data[(i_element + 3) * output_depth];

      const float* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const float* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const float* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const float* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_data[i_ch_in * output_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_in + 1) * output_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_in + 2) * output_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_in + 3) * output_depth];
      const int8_t* filter_4_int8 = &filter_data[(i_ch_in + 4) * output_depth];
      const int8_t* filter_5_int8 = &filter_data[(i_ch_in + 5) * output_depth];
      const int8_t* filter_6_int8 = &filter_data[(i_ch_in + 6) * output_depth];
      const int8_t* filter_7_int8 = &filter_data[(i_ch_in + 7) * output_depth];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[32] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[16], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[20], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[24], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_4row_8col_fp_IOHW_forint8w(&sum[28], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);


        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[8], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[9], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[10], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[11], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[12], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[13], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[14], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[15], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[16], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[17], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[18], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[19], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[20], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[21], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[22], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[23], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[24], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[25], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[26], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[27], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[28], output_activation_min), output_activation_max);
        *out_1++ += MIN(MAX(sum[29], output_activation_min), output_activation_max);
        *out_2++ += MIN(MAX(sum[30], output_activation_min), output_activation_max);
        *out_3++ += MIN(MAX(sum[31], output_activation_min), output_activation_max);
      }
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=8) {
      float* out_0 = &output_data[(num_elements - leftover_elements) * output_depth];

      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_data[i_ch_in * output_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_in + 1) * output_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_in + 2) * output_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_in + 3) * output_depth];
      const int8_t* filter_4_int8 = &filter_data[(i_ch_in + 4) * output_depth];
      const int8_t* filter_5_int8 = &filter_data[(i_ch_in + 5) * output_depth];
      const int8_t* filter_6_int8 = &filter_data[(i_ch_in + 6) * output_depth];
      const int8_t* filter_7_int8 = &filter_data[(i_ch_in + 7) * output_depth];
      float filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        float sum[8] = {};

        /* MAC computation */
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        filter_4 = (float)*filter_4_int8++; filter_5 = (float)*filter_5_int8++; filter_6 = (float)*filter_6_int8++; filter_7 = (float)*filter_7_int8++;
        mac_1row_8col_fp_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7);

        /* Accumulate partial sum into output data */
        *out_0++ += MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out_0++ += MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8w_partialCH(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;

  /* Initialize output data as 0 (assume bias == NULL) */
  for(i_element = 0; i_element < output_depth*num_elements; i_element++) {
    output_data[i_element] = 0;
  }

  for (i_element = 0; i_element/4 < num_elements/4; i_element+=4) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      float* out_0 = &output_data[i_element * output_depth];
      float* out_1 = &output_data[(i_element + 1) * output_depth];
      float* out_2 = &output_data[(i_element + 2) * output_depth];
      float* out_3 = &output_data[(i_element + 3) * output_depth];

      const float* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const float* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const float* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const float* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_sram[i_ch_in * first_k_channel];
      const int8_t* filter_1_int8 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const int8_t* filter_2_int8 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const int8_t* filter_3_int8 = &filter_sram[(i_ch_in + 3) * first_k_channel];
      float filter_0, filter_1, filter_2, filter_3;

      /* Compute weights in SRAM */
      uint16_t col_count_div8 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 3;
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
        assign_sum_to_pointwise_output_4row8col(out_0, out_1, out_2, out_3, sum, output_activation_min, output_activation_max);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }

      filter_0_int8 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1_int8 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2_int8 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3_int8 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div8 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 3;
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
        assign_sum_to_pointwise_output_4row8col(out_0, out_1, out_2, out_3, sum, output_activation_min, output_activation_max);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      float* out_0 = &output_data[(num_elements - leftover_elements) * output_depth];

      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const int8_t* filter_0_int8 = &filter_sram[i_ch_in * first_k_channel];
      const int8_t* filter_1_int8 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const int8_t* filter_2_int8 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const int8_t* filter_3_int8 = &filter_sram[(i_ch_in + 3) * first_k_channel];
      float filter_0, filter_1, filter_2, filter_3;

      /* Compute weights in SRAM */
      uint16_t col_count_div8 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 3;
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
        assign_sum_to_pointwise_output_1row8col(out_0, sum, output_activation_min, output_activation_max);
        out_0 += 8;
      }

      filter_0_int8 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1_int8 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2_int8 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3_int8 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div8 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 3;
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
        assign_sum_to_pointwise_output_1row8col(out_0, sum, output_activation_min, output_activation_max);
        out_0 += 8;
      }
    }

    leftover_elements--;
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp pointwise_conv_fp_4row4col_IOHW_int8w(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;

  /* Initialize output data as 0 (assume bias == NULL) */
  for(i_element = 0; i_element < output_depth*num_elements; i_element++) {
    output_data[i_element] = 0;
  }

  for (i_element = 0; i_element/4 < num_elements/4; i_element+=4) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      float* out_0 = &output_data[i_element * output_depth];
      float* out_1 = &output_data[(i_element + 1) * output_depth];
      float* out_2 = &output_data[(i_element + 2) * output_depth];
      float* out_3 = &output_data[(i_element + 3) * output_depth];

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
        assign_sum_to_pointwise_output_4row8col(out_0, out_1, out_2, out_3, sum, output_activation_min, output_activation_max);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      float* out_0 = &output_data[(num_elements - leftover_elements) * output_depth];

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
        assign_sum_to_pointwise_output_1row8col(out_0, sum, output_activation_min, output_activation_max);
        out_0 += 8;
      }
    }

    leftover_elements--;
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp pointwise_conv_fp_4row4col_int8w(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=4) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      /*float sum[16] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3]};*/
      float sum[16] = {};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      const float* input_1 = &input_data[(i_element + 1) * input_depth];
      const float* input_2 = &input_data[(i_element + 2) * input_depth];
      const float* input_3 = &input_data[(i_element + 3) * input_depth];

      const int8_t* filter_0_int8 = &filter_data[i_ch_out * input_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_out + 1) * input_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_out + 2) * input_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_out + 3) * input_depth];
      float filter_0, filter_1, filter_2, filter_3;

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_forint8w(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_forint8w(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_forint8w(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_4row_4col_fp_forint8w(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      out += output_depth - 3;
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      out += output_depth - 3;
      *out++ = MIN(MAX(sum[8], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[9], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[10], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[11], output_activation_min), output_activation_max);
      out += output_depth - 3;
      *out++ = MIN(MAX(sum[12], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[13], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[14], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[15], output_activation_min), output_activation_max);
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=4) {
      float* out = &output_data[(num_elements - leftover_elements) * output_depth + i_ch_out];
      float sum[4] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth];

      const int8_t* filter_0_int8 = &filter_data[i_ch_out * input_depth];
      const int8_t* filter_1_int8 = &filter_data[(i_ch_out + 1) * input_depth];
      const int8_t* filter_2_int8 = &filter_data[(i_ch_out + 2) * input_depth];
      const int8_t* filter_3_int8 = &filter_data[(i_ch_out + 3) * input_depth];
      float filter_0, filter_1, filter_2, filter_3;

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++;

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 

        filter_0 = (float)*filter_0_int8++; filter_1 = (float)*filter_1_int8++; filter_2 = (float)*filter_2_int8++; filter_3 = (float)*filter_3_int8++;
        mac_1row_4col_fp_forint8w(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_4row4col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=4) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[16] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3],
                       bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      const float* input_1 = &input_data[(i_element + 1) * input_depth];
      const float* input_2 = &input_data[(i_element + 2) * input_depth];
      const float* input_3 = &input_data[(i_element + 3) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_4row_4col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_4row_4col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_4row_4col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_4row_4col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; filter_2++; filter_3++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      out += output_depth - 3;
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
      out += output_depth - 3;
      *out++ = MIN(MAX(sum[8], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[9], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[10], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[11], output_activation_min), output_activation_max);
      out += output_depth - 3;
      *out++ = MIN(MAX(sum[12], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[13], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[14], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[15], output_activation_min), output_activation_max);
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=4) {
      float* out = &output_data[(num_elements - leftover_elements) * output_depth + i_ch_out];
      float sum[4] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++;
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[3], output_activation_min), output_activation_max);
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_4row2col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=2) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[8] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out], bias_data[i_ch_out + 1],
                     bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out], bias_data[i_ch_out + 1]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      const float* input_1 = &input_data[(i_element + 1) * input_depth];
      const float* input_2 = &input_data[(i_element + 2) * input_depth];
      const float* input_3 = &input_data[(i_element + 3) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_4row_2col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; 

        mac_4row_2col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; 

        mac_4row_2col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; 

        mac_4row_2col_fp(sum, input_0, input_1, input_2, input_3, filter_0, filter_1);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; filter_1++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      out += output_depth - 1;
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      out += output_depth - 1;
      *out++ = MIN(MAX(sum[4], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[5], output_activation_min), output_activation_max);
      out += output_depth - 1;
      *out++ = MIN(MAX(sum[6], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[7], output_activation_min), output_activation_max);
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out+=4) {
      float* out = &output_data[(num_elements - leftover_elements) * output_depth + i_ch_out];
      float sum[4] = {bias_data[i_ch_out], bias_data[i_ch_out + 1], bias_data[i_ch_out + 2], bias_data[i_ch_out + 3]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];
      const float* filter_1 = &filter_data[(i_ch_out + 1) * input_depth];
      const float* filter_2 = &filter_data[(i_ch_out + 2) * input_depth];
      const float* filter_3 = &filter_data[(i_ch_out + 3) * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++;
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; 

        mac_1row_4col_fp(sum, input_0, filter_0, filter_1, filter_2, filter_3);
        input_0++; 
        filter_0++; filter_1++; filter_2++; filter_3++; 
      }

      *out++ = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      *out = MIN(MAX(sum[3], output_activation_min), output_activation_max);
    }

    leftover_elements--;
  }
}

tinyengine_status_fp pointwise_conv_fp_4row1col(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  /*
  if (input_depth % 4 != 0 || input_depth % 2 != 0) {
    return PARAM_NO_SUPPORT;
  }
  */

  (void) input_height;
  (void) input_width;

  //const int channel_div4 = (input_depth >> 2);
  int i_element;

  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element+=4) {
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out++) {
      float* out = &output_data[i_element * output_depth + i_ch_out];
      float sum[4] = {bias_data[i_ch_out], bias_data[i_ch_out], 
                     bias_data[i_ch_out], bias_data[i_ch_out]};

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[i_element * input_depth];
      const float* input_1 = &input_data[(i_element + 1) * input_depth];
      const float* input_2 = &input_data[(i_element + 2) * input_depth];
      const float* input_3 = &input_data[(i_element + 3) * input_depth];

      const float* filter_0 = &filter_data[i_ch_out * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        mac_4row_1col_fp(sum, input_0, input_1, input_2, input_3, filter_0);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; 

        mac_4row_1col_fp(sum, input_0, input_1, input_2, input_3, filter_0);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; 

        mac_4row_1col_fp(sum, input_0, input_1, input_2, input_3, filter_0);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; 

        mac_4row_1col_fp(sum, input_0, input_1, input_2, input_3, filter_0);
        input_0++; input_1++; input_2++; input_3++;
        filter_0++; 
      }

      *out = MIN(MAX(sum[0], output_activation_min), output_activation_max);
      out += output_depth;
      *out = MIN(MAX(sum[1], output_activation_min), output_activation_max);
      out += output_depth;
      *out = MIN(MAX(sum[2], output_activation_min), output_activation_max);
      out += output_depth;
      *out = MIN(MAX(sum[3], output_activation_min), output_activation_max);
      out += output_depth;
    }
  }

  /* Handle left-over part */
  int leftover_elements = num_elements & 0x3;

  while (leftover_elements) { 
    int i_ch_out;

    for (i_ch_out = 0; i_ch_out < output_depth; i_ch_out++) {
      float sum = bias_data[i_ch_out];

      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const float* input_0 = &input_data[(num_elements - leftover_elements) * input_depth];
      const float* filter_0 = &filter_data[i_ch_out * input_depth];

      uint16_t col_count_div4 = (input_depth * DIM_KER_X * DIM_KER_Y) >> 2;

      while (col_count_div4--) {
        sum += *input_0++ * *filter_0++;
        sum += *input_0++ * *filter_0++;
        sum += *input_0++ * *filter_0++;
        sum += *input_0++ * *filter_0++;
      }

      output_data[(num_elements - leftover_elements) * output_depth + i_ch_out] = MIN(MAX(sum, output_activation_min), output_activation_max);
    }

    leftover_elements--;
  }
}