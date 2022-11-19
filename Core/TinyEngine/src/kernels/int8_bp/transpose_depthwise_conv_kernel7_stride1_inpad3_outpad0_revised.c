/* ----------------------------------------------------------------------
 * Name: transpose_depthwise_conv_kernel7_stride1_inpad3_outpad0_revised.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function.h"
#include "nnfunctions.h"
#define DIM_KER_X (7U)
#define DIM_KER_Y (7U)
#define STRIDE (1U)
#define IN_PAD (3U)
#define OUT_PAD (0U)

// The output resolution of this transposed conv operator will remain same as the input resolution. Thus, this operator adopts in-place depthwise conv which
// directly use the input data buffer as the output data buffer.
// partialCH is to enable sparse update.
tinyengine_status transpose_depthwise_conv_kernel7_stride1_inpad3_outpad0_revised_IOHW_partialCH(int8_t* input_output_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const int32_t* bias_data, 
                 int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const int32_t output_activation_min, const int32_t output_activation_max,
                 int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value) {
  int8_t* two_column_buffer = im2col_data;
  q31_t* tmp_output_buffer = norm_data;
  int8_t* input_output_data_start = input_output_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < input_width + 6; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: left and right regions
  for (i = 0; i < input_height; i++) {
    *two_column_buffer++ = pad_value; // left 1
    *two_column_buffer++ = pad_value; // left 2
    *two_column_buffer++ = pad_value; // left 3
    two_column_buffer += input_width; // skip middle
    *two_column_buffer++ = pad_value; // right 1
    *two_column_buffer++ = pad_value; // right 2
    *two_column_buffer++ = pad_value; // right 3
  }
  // Bottom region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < input_width + 6; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }

  /* Setup the input_output_data regions for HWC->CHW buffers */
  const int8_t* src;
  const int8_t* ksrc;
  int8_t ksrc_transposed[49];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width + 6) * 3;
    src = input_output_data_start;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 3;;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }

      two_column_buffer += 3;
    }

    // !! Critical part for transposed depthwise conv !! (Different from the implementation of general transposed conv)
    /* Transpose filter data */ 
    if (c < first_k_channel) {
      ksrc = filter_sram++;
    }
    else {
      ksrc = filter_flash++;
    }
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[48 - i] = *ksrc;

      if (c < first_k_channel) {
        ksrc += first_k_channel;
      }
      else {
        ksrc += input_depth - first_k_channel;
      }
    }

    int8_t* two_column_buffer_start = im2col_data;
    q31_t* tmp_output = tmp_output_buffer;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        int32_t sum_0 = 0;
        int32_t sum_1 = 0;
        transpose_depthwise_mac_kernel7_2row_uniweight(&sum_0, &sum_1, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        tmp_output[(i * output_width + j) * output_depth] = sum_0;
        tmp_output[(i * output_width + j + 1) * output_depth] = sum_1;

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        int32_t sum_0 = 0;
        transpose_depthwise_mac_kernel7_1row_uniweight(&sum_0, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        tmp_output[(i * output_width + output_width - 1) * output_depth] = sum_0;

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 6;
    }

    input_output_data_start++;
    tmp_output_buffer++;
  }

  /* Output Normalization */
  int num_elements = output_height * output_width;
  int i_element;

  int8_t* inplace_output = input_output_data;
  q31_t* tmp_output = norm_data;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_output_depth;
    q31_t out_max = 0;
    
    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max = MAX(out_max, abs(*tmp_output));
      tmp_output++;
    }

    tmp_output = norm_data;
    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *inplace_output++ = (q7_t) ((float)*tmp_output/(out_max/127));
      tmp_output++;
    }

    norm_data += output_depth;
  }

  /* Return to application */
  return STATE_SUCCESS;
} 


// The output resolution of this transposed conv operator will remain same as the input resolution. Thus, this operator adopts in-place depthwise conv which
// directly use the input data buffer as the output data buffer.
tinyengine_status transpose_depthwise_conv_kernel7_stride1_inpad3_outpad0_revised_IOHW(int8_t* input_output_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const int32_t* bias_data, 
                 int8_t* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const int32_t output_activation_min, const int32_t output_activation_max,
                 int8_t* im2col_data, q31_t* norm_data, const uint16_t batches, const int8_t pad_value) {
  int8_t* two_column_buffer = im2col_data;
  q31_t* tmp_output_buffer = norm_data;
  int8_t* input_output_data_start = input_output_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < input_width + 6; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: left and right regions
  for (i = 0; i < input_height; i++) {
    *two_column_buffer++ = pad_value; // left 1
    *two_column_buffer++ = pad_value; // left 2
    *two_column_buffer++ = pad_value; // left 3
    two_column_buffer += input_width; // skip middle
    *two_column_buffer++ = pad_value; // right 1
    *two_column_buffer++ = pad_value; // right 2
    *two_column_buffer++ = pad_value; // right 3
  }
  // Bottom region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < input_width + 6; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }

  /* Setup the input_output_data regions for HWC->CHW buffers */
  const int8_t* src;
  const int8_t* ksrc;
  int8_t ksrc_transposed[49];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width + 6) * 3;
    src = input_output_data_start;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 3;;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }

      two_column_buffer += 3;
    }

    // !! Critical part for transposed depthwise conv !! (Different from the implementation of general transposed conv)
    /* Transpose filter data */ 
    ksrc = filter_data++;
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[48 - i] = *ksrc;
      ksrc += input_depth;
    }

    int8_t* two_column_buffer_start = im2col_data;
    q31_t* tmp_output = tmp_output_buffer;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        int32_t sum_0 = 0;
        int32_t sum_1 = 0;
        transpose_depthwise_mac_kernel7_2row_uniweight(&sum_0, &sum_1, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        tmp_output[(i * output_width + j) * output_depth] = sum_0;
        tmp_output[(i * output_width + j + 1) * output_depth] = sum_1;

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        int32_t sum_0 = 0;
        transpose_depthwise_mac_kernel7_1row_uniweight(&sum_0, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        tmp_output[(i * output_width + output_width - 1) * output_depth] = sum_0;

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 6;
    }

    input_output_data_start++;
    tmp_output_buffer++;
  }

  /* Output Normalization */
  int num_elements = output_height * output_width;
  int i_element;

  int8_t* inplace_output = input_output_data;
  q31_t* tmp_output = norm_data;

  for (i_element = 0; i_element < num_elements; i_element++) {
    int i_output_depth;
    q31_t out_max = 0;
    
    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      out_max = MAX(out_max, abs(*tmp_output));
      tmp_output++;
    }

    tmp_output = norm_data;
    for (i_output_depth = 0; i_output_depth < output_depth; i_output_depth++) {
      *inplace_output++ = (q7_t) ((float)*tmp_output/(out_max/127));
      tmp_output++;
    }

    norm_data += output_depth;
  }
  
  /* Return to application */
  return STATE_SUCCESS;
} 
