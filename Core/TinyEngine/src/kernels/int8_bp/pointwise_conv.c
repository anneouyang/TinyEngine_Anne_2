/* ----------------------------------------------------------------------
 * Name: pointwise_conv.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"
#include "nnfunctions.h"
#include "arm_nnfunctions.h"
#include "img2col_element.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status pointwise_conv_1row10col_10inputdepth_IOHW_int8w(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const q31_t* bias_data, 
                 q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 int16_t* im2col_data, q31_t* norm_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  q7_t* out = output_data;

  int i_element;
  const int num_elements = output_height * output_width;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_ch_in;

    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=10) {
      /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
      const q7_t* input_0 = &input_data[i_element * input_depth];

      const q7_t* filter_0 = &filter_data[i_ch_in * output_depth];
      const q7_t* filter_1 = &filter_data[(i_ch_in + 1) * output_depth];
      const q7_t* filter_2 = &filter_data[(i_ch_in + 2) * output_depth];
      const q7_t* filter_3 = &filter_data[(i_ch_in + 3) * output_depth];
      const q7_t* filter_4 = &filter_data[(i_ch_in + 4) * output_depth];
      const q7_t* filter_5 = &filter_data[(i_ch_in + 5) * output_depth];
      const q7_t* filter_6 = &filter_data[(i_ch_in + 6) * output_depth];
      const q7_t* filter_7 = &filter_data[(i_ch_in + 7) * output_depth];
      const q7_t* filter_8 = &filter_data[(i_ch_in + 8) * output_depth];
      const q7_t* filter_9 = &filter_data[(i_ch_in + 9) * output_depth];

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        q31_t sum[8] = {};

        mac_1row_10col_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        mac_1row_10col_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, filter_7, filter_8, filter_9);
        filter_0++; filter_1++; filter_2++; filter_3++; filter_4++; filter_5++; filter_6++; filter_7++; filter_8++; filter_9++; 

        *out++ = (q7_t) MIN(MAX(sum[0], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[1], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[2], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[3], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[4], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[5], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[6], output_activation_min), output_activation_max);
        *out++ = (q7_t) MIN(MAX(sum[7], output_activation_min), output_activation_max);
      }
    }
  }

  /* Return to application */
  return STATE_SUCCESS;
}

tinyengine_status pointwise_conv_4row4col_IOHW_int8input_int8w(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_data, const q31_t* bias_data, 
                 q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 int16_t* im2col_data, q31_t* norm_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;
  q31_t* tmp_output_buffer = norm_data;
  q7_t* output = output_data;

  // Since each kernel in pointwise conv only does conv operations along with the channel direction (doesn't care x or y directions like general 3x3 conv), 
  // so here we loop num_elements (= output_height * output_width).
  for (i_element = 0; i_element/4 < num_elements/4; i_element+=4) {
    // In int8 only backward propagation, we will need to do output normalization (e.g., Lines 187 - 230), so we use an output buffer (named norm_data) 
    // for storing the temporary int32 output data before finishing normalization.
    /* Initialize output data as 0 (assume bias == NULL) */
    int i;
    for(i = 0; i < output_depth * 4; i+=4) {
      tmp_output_buffer[i] = 0;
      tmp_output_buffer[i + 1] = 0;
      tmp_output_buffer[i + 2] = 0;
      tmp_output_buffer[i + 3] = 0;
    }

    int i_ch_in;

    // Generally for OHWI (NHWC), here we should loop output_depth (can refer to "TinyEngine/src/kernels/convolve_1x1_s8.c" to see the implementation for 
    // general pointwise conv). But since in the backward propagation the kernel layout is IOHW (CNHW), here we loop input_depth for a better sequential 
    // data access pattern.
    // P.S. If the implementation of IOHW (CNHW) is hard to understand, you can first look into the implementation of general pointwise conv "convolve_1x1_s8.c" 
    // and then compare the difference between these two files. This might be helpful. Besides, you can also refer to "convolve_1x1_s8.c" for how to enable 
    // SIMD on MCUs.
    for (i_ch_in = 0; i_ch_in < input_depth; i_ch_in+=4) {
      q31_t* out_0 = tmp_output_buffer;
      q31_t* out_1 = &tmp_output_buffer[output_depth * 1];
      q31_t* out_2 = &tmp_output_buffer[output_depth * 2];
      q31_t* out_3 = &tmp_output_buffer[output_depth * 3];

      const q7_t* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const q7_t* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const q7_t* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const q7_t* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const q7_t* filter_0 = &filter_data[i_ch_in * output_depth];
      const q7_t* filter_1 = &filter_data[(i_ch_in + 1) * output_depth];
      const q7_t* filter_2 = &filter_data[(i_ch_in + 2) * output_depth];
      const q7_t* filter_3 = &filter_data[(i_ch_in + 3) * output_depth];

      
      q15_t* runtime_buf = (q15_t*) im2col_data;

      // use variables
      q31_t in_q7x4;
      q31_t in_q15x2_1;
      q31_t in_q15x2_2;
      q31_t out_q15x2_1;
      q31_t out_q15x2_2;

      q7_q15_reordered_ele(input_0, runtime_buf);
      input_0 -= 4;
      q7_q15_reordered_ele(input_1, runtime_buf);
      input_1 -= 4;
      q7_q15_reordered_ele(input_2, runtime_buf);
      input_2 -= 4;
      q7_q15_reordered_ele(input_3, runtime_buf);
      input_3 -= 4;
      runtime_buf -= 16;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[32] = {};

        /* MAC computation */

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[0], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[4], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[8], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[12], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[16], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[20], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[24], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[28], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        
        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_4row8col_int8(out_0, out_1, out_2, out_3, sum);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }
    }

    /* Output Normalization */
    q31_t* tmp_out_0 = tmp_output_buffer;
    q31_t* tmp_out_1 = &tmp_output_buffer[output_depth * 1];
    q31_t* tmp_out_2 = &tmp_output_buffer[output_depth * 2];
    q31_t* tmp_out_3 = &tmp_output_buffer[output_depth * 3];
    q31_t out_max_0 = 0; q31_t out_max_1 = 0; q31_t out_max_2 = 0; q31_t out_max_3 = 0; 
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, abs(*tmp_out_0));
      tmp_out_0++;

      out_max_1 = MAX(out_max_1, abs(*tmp_out_1));
      tmp_out_1++;

      out_max_2 = MAX(out_max_2, abs(*tmp_out_2));
      tmp_out_2++;

      out_max_3 = MAX(out_max_3, abs(*tmp_out_3));
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
      *out_0++ = (q7_t) ((float)*tmp_out_0/(out_max_0/127));
      tmp_out_0++;

      *out_1++ = (q7_t) ((float)*tmp_out_1/(out_max_1/127));
      tmp_out_1++;

      *out_2++ = (q7_t) ((float)*tmp_out_2/(out_max_2/127));
      tmp_out_2++;

      *out_3++ = (q7_t) ((float)*tmp_out_3/(out_max_3/127));
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
      q31_t* out_0 = tmp_output_buffer;

      const q7_t* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const q7_t* filter_0 = &filter_data[i_ch_in * output_depth];
      const q7_t* filter_1 = &filter_data[(i_ch_in + 1) * output_depth];
      const q7_t* filter_2 = &filter_data[(i_ch_in + 2) * output_depth];
      const q7_t* filter_3 = &filter_data[(i_ch_in + 3) * output_depth];

      q15_t* runtime_buf = (q15_t*) im2col_data;

      // use variables
      q31_t in_q7x4;
      q31_t in_q15x2_1;
      q31_t in_q15x2_2;
      q31_t out_q15x2_1;
      q31_t out_q15x2_2;

      q7_q15_reordered_ele(input_0, runtime_buf);
      input_0 -= 4;
      runtime_buf -= 16;

      uint16_t col_count_div8 = (output_depth * DIM_KER_X * DIM_KER_Y) >> 3;

      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[8] = {};

        /* MAC computation */
        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[0], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[1], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[2], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[3], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[4], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[5], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[6], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        mac_1row_4col_IOHW_forint8w_s8_fpreq(&sum[7], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_1row8col_int8(out_0, sum);
        out_0 += 8;
      }
    }

    /* Output Normalization */
    q31_t* tmp_out_0 = tmp_output_buffer;
    q31_t out_max_0 = 0;
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, abs(*tmp_out_0));
      tmp_out_0++;
    }

    q7_t* out_0 = &output[(num_elements - leftover_elements) * output_depth];
    tmp_out_0 = tmp_output_buffer;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *out_0++ = (q7_t) ((float)*tmp_out_0/(out_max_0/127));
      tmp_out_0++;
    }

    leftover_elements--;
  }

  /* Return to application */
  return STATE_SUCCESS;
}

// partialCH is to enable sparse update.
tinyengine_status pointwise_conv_4row4col_IOHW_int8input_int8w_partialCH(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_sram, const q7_t* filter_flash, const uint16_t first_k_channel, const q31_t* bias_data, 
                 q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 int16_t* im2col_data, q31_t* norm_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;
  q31_t* tmp_output_buffer = norm_data;
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
      q31_t* out_0 = tmp_output_buffer;
      q31_t* out_1 = &tmp_output_buffer[output_depth * 1];
      q31_t* out_2 = &tmp_output_buffer[output_depth * 2];
      q31_t* out_3 = &tmp_output_buffer[output_depth * 3];

      const q7_t* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const q7_t* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const q7_t* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const q7_t* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const q7_t* filter_0 = &filter_sram[i_ch_in * first_k_channel];
      const q7_t* filter_1 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const q7_t* filter_2 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const q7_t* filter_3 = &filter_sram[(i_ch_in + 3) * first_k_channel];

      q15_t* runtime_buf = (q15_t*) im2col_data;

      // use variables
      q31_t in_q7x4;
      q31_t in_q15x2_1;
      q31_t in_q15x2_2;
      q31_t out_q15x2_1;
      q31_t out_q15x2_2;

      q7_q15_reordered_ele(input_0, runtime_buf);
      input_0 -= 4;
      q7_q15_reordered_ele(input_1, runtime_buf);
      input_1 -= 4;
      q7_q15_reordered_ele(input_2, runtime_buf);
      input_2 -= 4;
      q7_q15_reordered_ele(input_3, runtime_buf);
      input_3 -= 4;
      runtime_buf -= 16;

      /* Compute weights in SRAM */
      uint16_t col_count_div8 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 3;
      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[32] = {};

        /* MAC computation */
        // mac_4row_4col_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[16], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[20], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[24], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[28], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[0], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[4], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[8], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[12], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[16], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[20], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[24], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[28], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_4row8col_int8(out_0, out_1, out_2, out_3, sum);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }

      filter_0 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div8 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 3;
      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[32] = {};

        /* MAC computation */
        // mac_4row_4col_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[16], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[20], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[24], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;
        // mac_4row_4col_IOHW_forint8w(&sum[28], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        // filter_0++; filter_1++; filter_2++; filter_3++;

        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[0], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[4], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[8], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[12], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[16], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[20], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[24], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w_s8_fpreq(&sum[28], runtime_buf, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_4row8col_int8(out_0, out_1, out_2, out_3, sum);
        out_0 += 8; out_1 += 8; out_2 += 8; out_3 += 8;
      }
    }

    /* Output Normalization */
    q31_t* tmp_out_0 = tmp_output_buffer;
    q31_t* tmp_out_1 = &tmp_output_buffer[output_depth * 1];
    q31_t* tmp_out_2 = &tmp_output_buffer[output_depth * 2];
    q31_t* tmp_out_3 = &tmp_output_buffer[output_depth * 3];
    q31_t out_max_0 = 0; q31_t out_max_1 = 0; q31_t out_max_2 = 0; q31_t out_max_3 = 0; 
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, abs(*tmp_out_0));
      tmp_out_0++;

      out_max_1 = MAX(out_max_1, abs(*tmp_out_1));
      tmp_out_1++;

      out_max_2 = MAX(out_max_2, abs(*tmp_out_2));
      tmp_out_2++;

      out_max_3 = MAX(out_max_3, abs(*tmp_out_3));
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
      *out_0++ = (q7_t) ((float)*tmp_out_0/(out_max_0/127));
      tmp_out_0++;

      *out_1++ = (q7_t) ((float)*tmp_out_1/(out_max_1/127));
      tmp_out_1++;

      *out_2++ = (q7_t) ((float)*tmp_out_2/(out_max_2/127));
      tmp_out_2++;

      *out_3++ = (q7_t) ((float)*tmp_out_3/(out_max_3/127));
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
      q31_t* out_0 = tmp_output_buffer;

      const q7_t* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const q7_t* filter_0 = &filter_sram[i_ch_in * first_k_channel];
      const q7_t* filter_1 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const q7_t* filter_2 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const q7_t* filter_3 = &filter_sram[(i_ch_in + 3) * first_k_channel];

      /* Compute weights in SRAM */
      uint16_t col_count_div8 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 3;
      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[8] = {};

        /* MAC computation */
        mac_1row_4col_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_1row8col_int8(out_0, sum);
        out_0 += 8;
      }

      filter_0 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div8 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 3;
      while (col_count_div8--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[8] = {};

        /* MAC computation */
        mac_1row_4col_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[4], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[5], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[6], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[7], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_1row8col_int8(out_0, sum);
        out_0 += 8;
      }
    }

    /* Output Normalization */
    q31_t* tmp_out_0 = tmp_output_buffer;
    q31_t out_max_0 = 0;
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, abs(*tmp_out_0));
      tmp_out_0++;
    }

    q7_t* out_0 = &output[(num_elements - leftover_elements) * output_depth];
    tmp_out_0 = tmp_output_buffer;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *out_0++ = (q7_t) ((float)*tmp_out_0/(out_max_0/127));
      tmp_out_0++;
    }

    leftover_elements--;
  }

  /* Return to application */
  return STATE_SUCCESS;
}

// partialCH is to enable sparse update.
tinyengine_status pointwise_conv_4row4col_4innercol_IOHW_int8input_int8w_partialCH(const q7_t* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const q7_t* filter_sram, const q7_t* filter_flash, const uint16_t first_k_channel, const q31_t* bias_data, 
                 q7_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const q31_t output_activation_min, const q31_t output_activation_max,
                 int16_t* im2col_data, q31_t* norm_data, const uint16_t batches) {
  (void) input_height;
  (void) input_width;

  int i_element;
  const int num_elements = output_height * output_width;
  q31_t* tmp_output_buffer = norm_data;
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
      q31_t* out_0 = tmp_output_buffer;
      q31_t* out_1 = &tmp_output_buffer[output_depth * 1];
      q31_t* out_2 = &tmp_output_buffer[output_depth * 2];
      q31_t* out_3 = &tmp_output_buffer[output_depth * 3];

      const q7_t* input_0 = &input_data[i_element * input_depth + i_ch_in];
      const q7_t* input_1 = &input_data[(i_element + 1) * input_depth + i_ch_in];
      const q7_t* input_2 = &input_data[(i_element + 2) * input_depth + i_ch_in];
      const q7_t* input_3 = &input_data[(i_element + 3) * input_depth + i_ch_in];

      const q7_t* filter_0 = &filter_sram[i_ch_in * first_k_channel];
      const q7_t* filter_1 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const q7_t* filter_2 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const q7_t* filter_3 = &filter_sram[(i_ch_in + 3) * first_k_channel];

      /* Compute weights in SRAM */
      uint16_t col_count_div4 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 2;
      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[16] = {};

        /* MAC computation */
        mac_4row_4col_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_4row4col_int8(out_0, out_1, out_2, out_3, sum);
        out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4;
      }

      filter_0 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div4 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 2;
      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[16] = {};

        /* MAC computation */
        mac_4row_4col_IOHW_forint8w(&sum[0], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w(&sum[4], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w(&sum[8], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_4row_4col_IOHW_forint8w(&sum[12], input_0, input_1, input_2, input_3, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_4row4col_int8(out_0, out_1, out_2, out_3, sum);
        out_0 += 4; out_1 += 4; out_2 += 4; out_3 += 4;
      }
    }

    /* Output Normalization */
    q31_t* tmp_out_0 = tmp_output_buffer;
    q31_t* tmp_out_1 = &tmp_output_buffer[output_depth * 1];
    q31_t* tmp_out_2 = &tmp_output_buffer[output_depth * 2];
    q31_t* tmp_out_3 = &tmp_output_buffer[output_depth * 3];
    q31_t out_max_0 = 0; q31_t out_max_1 = 0; q31_t out_max_2 = 0; q31_t out_max_3 = 0; 
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, abs(*tmp_out_0));
      tmp_out_0++;

      out_max_1 = MAX(out_max_1, abs(*tmp_out_1));
      tmp_out_1++;

      out_max_2 = MAX(out_max_2, abs(*tmp_out_2));
      tmp_out_2++;

      out_max_3 = MAX(out_max_3, abs(*tmp_out_3));
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
      *out_0++ = (q7_t) ((float)*tmp_out_0/(out_max_0/127));
      tmp_out_0++;

      *out_1++ = (q7_t) ((float)*tmp_out_1/(out_max_1/127));
      tmp_out_1++;

      *out_2++ = (q7_t) ((float)*tmp_out_2/(out_max_2/127));
      tmp_out_2++;

      *out_3++ = (q7_t) ((float)*tmp_out_3/(out_max_3/127));
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
      q31_t* out_0 = tmp_output_buffer;

      const q7_t* input_0 = &input_data[(num_elements - leftover_elements) * input_depth + i_ch_in];

      const q7_t* filter_0 = &filter_sram[i_ch_in * first_k_channel];
      const q7_t* filter_1 = &filter_sram[(i_ch_in + 1) * first_k_channel];
      const q7_t* filter_2 = &filter_sram[(i_ch_in + 2) * first_k_channel];
      const q7_t* filter_3 = &filter_sram[(i_ch_in + 3) * first_k_channel];

      /* Compute weights in SRAM */
      uint16_t col_count_div4 = (first_k_channel * DIM_KER_X * DIM_KER_Y) >> 2;
      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[4] = {};

        /* MAC computation */
        mac_1row_4col_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_1row4col_int8(out_0, sum);
        out_0 += 4;
      }

      filter_0 = &filter_flash[i_ch_in * (output_depth - first_k_channel)];
      filter_1 = &filter_flash[(i_ch_in + 1) * (output_depth - first_k_channel)];
      filter_2 = &filter_flash[(i_ch_in + 2) * (output_depth - first_k_channel)];
      filter_3 = &filter_flash[(i_ch_in + 3) * (output_depth - first_k_channel)];

      /* Compute weights in FLASH */
      col_count_div4 = ((output_depth - first_k_channel) * DIM_KER_X * DIM_KER_Y) >> 2;
      while (col_count_div4--) {
        /* Initialize partial sum (assume bias == NULL) */
        q31_t sum[4] = {};

        /* MAC computation */
        mac_1row_4col_IOHW_forint8w(&sum[0], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[1], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[2], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;
        mac_1row_4col_IOHW_forint8w(&sum[3], input_0, filter_0, filter_1, filter_2, filter_3);
        filter_0++; filter_1++; filter_2++; filter_3++;

        /* Accumulate partial sum into output data */
        assign_sum_to_pointwise_tmp_output_buffer_1row4col_int8(out_0, sum);
        out_0 += 4;
      }
    }

    /* Output Normalization */
    q31_t* tmp_out_0 = tmp_output_buffer;
    q31_t out_max_0 = 0;
    int i_output_depth;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max_0 = MAX(out_max_0, abs(*tmp_out_0));
      tmp_out_0++;
    }

    q7_t* out_0 = &output[(num_elements - leftover_elements) * output_depth];
    tmp_out_0 = tmp_output_buffer;

    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *out_0++ = (q7_t) ((float)*tmp_out_0/(out_max_0/127));
      tmp_out_0++;
    }

    leftover_elements--;
  }

  /* Return to application */
  return STATE_SUCCESS;
}
