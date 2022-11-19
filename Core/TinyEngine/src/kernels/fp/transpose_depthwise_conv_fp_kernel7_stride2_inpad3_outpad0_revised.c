/* ----------------------------------------------------------------------
 * Name: transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (7U)
#define DIM_KER_Y (7U)
#define STRIDE (2U)
#define IN_PAD (3U)
#define OUT_PAD (0U)

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised_IOHW_int8w_partialCH(float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_sram, const int8_t* filter_flash, const uint16_t first_k_channel, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region
  for (i = 0; i < input_width * 2 + 5; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: Pad the size of (input_height * 2) * (input_width * 2 + 2)
  for (i = 0; i < input_height; i++) {
    // First type of middle
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    for (j = 0; j < input_width; j++) {
      *two_column_buffer = pad_value;
      two_column_buffer += 2;
    }
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;

    // Second type of middle
    for (j = 0; j < input_width * 2 + 5; j++) {
      *two_column_buffer++ = pad_value;
    }
  }
  // Bottom region
  for (i = 0; i < input_width * 2 + 5; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }


  /* Setup the input_data regions for HWC->CHW buffers */
  const float* src;
  const int8_t* ksrc;
  float ksrc_transposed[49];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width * 2 + 5) * 3;
    src = input_data;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 3;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer = *src;
        two_column_buffer += 2;
        src += input_depth;
      }

      two_column_buffer += input_width * 2 + 7;
    }

    // Transpose filter data
    if (c < first_k_channel) {
      ksrc = filter_sram++;
    }
    else {
      ksrc = filter_flash++;
    }
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[48 - i] = (float)*ksrc;

      if (c < first_k_channel) {
        ksrc += first_k_channel;
      }
      else {
        ksrc += input_depth - first_k_channel;
      }
    }

    float* out = output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        float sum_1 = 0.0f;
        transpose_depthwise_mac_kernel7_2row_fp_uniweight(&sum_0, &sum_1, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + j) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);
        out[(i * output_width + j + 1) * output_depth] = MIN(MAX(sum_1, output_activation_min), output_activation_max);

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        transpose_depthwise_mac_kernel7_1row_fp_uniweight(&sum_0, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + output_width - 1) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 6;
    }

    bias_data++;
    input_data++;
    output_data++;
  }

  /* Return to application */
  return STATE_SUCCESS_fp;
} 

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised_IOHW_int8w(float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const int8_t* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region
  for (i = 0; i < input_width * 2 + 5; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: Pad the size of (input_height * 2) * (input_width * 2 + 2)
  for (i = 0; i < input_height; i++) {
    // First type of middle
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    for (j = 0; j < input_width; j++) {
      *two_column_buffer = pad_value;
      two_column_buffer += 2;
    }
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;

    // Second type of middle
    for (j = 0; j < input_width * 2 + 5; j++) {
      *two_column_buffer++ = pad_value;
    }
  }
  // Bottom region
  for (i = 0; i < input_width * 2 + 5; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }


  /* Setup the input_data regions for HWC->CHW buffers */
  const float* src;
  const int8_t* ksrc;
  float ksrc_transposed[49];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width * 2 + 5) * 3;
    src = input_data;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 3;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer = *src;
        two_column_buffer += 2;
        src += input_depth;
      }

      two_column_buffer += input_width * 2 + 7;
    }

    // Transpose filter data
    ksrc = filter_data++;
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[48 - i] = (float)*ksrc;
      ksrc += input_depth;
    }

    float* out = output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        float sum_1 = 0.0f;
        transpose_depthwise_mac_kernel7_2row_fp_uniweight(&sum_0, &sum_1, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + j) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);
        out[(i * output_width + j + 1) * output_depth] = MIN(MAX(sum_1, output_activation_min), output_activation_max);

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        // We assume bias_data as zeros.
        float sum_0 = 0.0f;
        transpose_depthwise_mac_kernel7_1row_fp_uniweight(&sum_0, two_column_buffer, ksrc_transposed, input_width, STRIDE, IN_PAD, OUT_PAD);
        out[(i * output_width + output_width - 1) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 6;
    }

    bias_data++;
    input_data++;
    output_data++;
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
} 

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride2_inpad3_outpad0_revised(float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region
  for (i = 0; i < input_width * 2 + 5; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: Pad the size of (input_height * 2) * (input_width * 2 + 2)
  for (i = 0; i < input_height; i++) {
    // First type of middle
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    for (j = 0; j < input_width; j++) {
      *two_column_buffer = pad_value;
      two_column_buffer += 2;
    }
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;

    // Second type of middle
    for (j = 0; j < input_width * 2 + 5; j++) {
      *two_column_buffer++ = pad_value;
    }
  }
  // Bottom region
  for (i = 0; i < input_width * 2 + 5; i++) {
    *two_column_buffer++ = pad_value;
    *two_column_buffer++ = pad_value;
  }


  /* Setup the input_data regions for HWC->CHW buffers */
  const float* src;
  const float* ksrc = filter_data;
  float ksrc_transposed[49];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width * 2 + 5) * 3;
    src = input_data;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 3;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer = *src;
        two_column_buffer += 2;
        src += input_depth;
      }

      two_column_buffer += input_width * 2 + 7;
    }

    // Transpose filter data
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[48 - i] = *ksrc++;
    }

    float* out = output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        float sum_0 = *bias_data;
        float sum_1 = *bias_data;

        sum_0 += two_column_buffer[0] * ksrc_transposed[0];
        sum_1 += two_column_buffer[1] * ksrc_transposed[0];
        sum_0 += two_column_buffer[1] * ksrc_transposed[1];
        sum_1 += two_column_buffer[2] * ksrc_transposed[1];
        sum_0 += two_column_buffer[2] * ksrc_transposed[2];
        sum_1 += two_column_buffer[3] * ksrc_transposed[2];
        sum_0 += two_column_buffer[3] * ksrc_transposed[3];
        sum_1 += two_column_buffer[4] * ksrc_transposed[3];
        sum_0 += two_column_buffer[4] * ksrc_transposed[4];
        sum_1 += two_column_buffer[5] * ksrc_transposed[4];
        sum_0 += two_column_buffer[5] * ksrc_transposed[5];
        sum_1 += two_column_buffer[6] * ksrc_transposed[5];
        sum_0 += two_column_buffer[6] * ksrc_transposed[6];
        sum_1 += two_column_buffer[7] * ksrc_transposed[6];
        two_column_buffer += input_width * 2 + 5;
        
        sum_0 += two_column_buffer[0] * ksrc_transposed[7];
        sum_1 += two_column_buffer[1] * ksrc_transposed[7];
        sum_0 += two_column_buffer[1] * ksrc_transposed[8];
        sum_1 += two_column_buffer[2] * ksrc_transposed[8];
        sum_0 += two_column_buffer[2] * ksrc_transposed[9];
        sum_1 += two_column_buffer[3] * ksrc_transposed[9];
        sum_0 += two_column_buffer[3] * ksrc_transposed[10];
        sum_1 += two_column_buffer[4] * ksrc_transposed[10];
        sum_0 += two_column_buffer[4] * ksrc_transposed[11];
        sum_1 += two_column_buffer[5] * ksrc_transposed[11];
        sum_0 += two_column_buffer[5] * ksrc_transposed[12];
        sum_1 += two_column_buffer[6] * ksrc_transposed[12];
        sum_0 += two_column_buffer[6] * ksrc_transposed[13];
        sum_1 += two_column_buffer[7] * ksrc_transposed[13];
        two_column_buffer += input_width * 2 + 5;
        
        sum_0 += two_column_buffer[0] * ksrc_transposed[14];
        sum_1 += two_column_buffer[1] * ksrc_transposed[14];
        sum_0 += two_column_buffer[1] * ksrc_transposed[15];
        sum_1 += two_column_buffer[2] * ksrc_transposed[15];
        sum_0 += two_column_buffer[2] * ksrc_transposed[16];
        sum_1 += two_column_buffer[3] * ksrc_transposed[16];
        sum_0 += two_column_buffer[3] * ksrc_transposed[17];
        sum_1 += two_column_buffer[4] * ksrc_transposed[17];
        sum_0 += two_column_buffer[4] * ksrc_transposed[18];
        sum_1 += two_column_buffer[5] * ksrc_transposed[18];
        sum_0 += two_column_buffer[5] * ksrc_transposed[19];
        sum_1 += two_column_buffer[6] * ksrc_transposed[19];
        sum_0 += two_column_buffer[6] * ksrc_transposed[20];
        sum_1 += two_column_buffer[7] * ksrc_transposed[20];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[21];
        sum_1 += two_column_buffer[1] * ksrc_transposed[21];
        sum_0 += two_column_buffer[1] * ksrc_transposed[22];
        sum_1 += two_column_buffer[2] * ksrc_transposed[22];
        sum_0 += two_column_buffer[2] * ksrc_transposed[23];
        sum_1 += two_column_buffer[3] * ksrc_transposed[23];
        sum_0 += two_column_buffer[3] * ksrc_transposed[24];
        sum_1 += two_column_buffer[4] * ksrc_transposed[24];
        sum_0 += two_column_buffer[4] * ksrc_transposed[25];
        sum_1 += two_column_buffer[5] * ksrc_transposed[25];
        sum_0 += two_column_buffer[5] * ksrc_transposed[26];
        sum_1 += two_column_buffer[6] * ksrc_transposed[26];
        sum_0 += two_column_buffer[6] * ksrc_transposed[27];
        sum_1 += two_column_buffer[7] * ksrc_transposed[27];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[28];
        sum_1 += two_column_buffer[1] * ksrc_transposed[28];
        sum_0 += two_column_buffer[1] * ksrc_transposed[29];
        sum_1 += two_column_buffer[2] * ksrc_transposed[29];
        sum_0 += two_column_buffer[2] * ksrc_transposed[30];
        sum_1 += two_column_buffer[3] * ksrc_transposed[30];
        sum_0 += two_column_buffer[3] * ksrc_transposed[31];
        sum_1 += two_column_buffer[4] * ksrc_transposed[31];
        sum_0 += two_column_buffer[4] * ksrc_transposed[32];
        sum_1 += two_column_buffer[5] * ksrc_transposed[32];
        sum_0 += two_column_buffer[5] * ksrc_transposed[33];
        sum_1 += two_column_buffer[6] * ksrc_transposed[33];
        sum_0 += two_column_buffer[6] * ksrc_transposed[34];
        sum_1 += two_column_buffer[7] * ksrc_transposed[34];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[35];
        sum_1 += two_column_buffer[1] * ksrc_transposed[35];
        sum_0 += two_column_buffer[1] * ksrc_transposed[36];
        sum_1 += two_column_buffer[2] * ksrc_transposed[36];
        sum_0 += two_column_buffer[2] * ksrc_transposed[37];
        sum_1 += two_column_buffer[3] * ksrc_transposed[37];
        sum_0 += two_column_buffer[3] * ksrc_transposed[38];
        sum_1 += two_column_buffer[4] * ksrc_transposed[38];
        sum_0 += two_column_buffer[4] * ksrc_transposed[39];
        sum_1 += two_column_buffer[5] * ksrc_transposed[39];
        sum_0 += two_column_buffer[5] * ksrc_transposed[40];
        sum_1 += two_column_buffer[6] * ksrc_transposed[40];
        sum_0 += two_column_buffer[6] * ksrc_transposed[41];
        sum_1 += two_column_buffer[7] * ksrc_transposed[41];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[42];
        sum_1 += two_column_buffer[1] * ksrc_transposed[42];
        sum_0 += two_column_buffer[1] * ksrc_transposed[43];
        sum_1 += two_column_buffer[2] * ksrc_transposed[43];
        sum_0 += two_column_buffer[2] * ksrc_transposed[44];
        sum_1 += two_column_buffer[3] * ksrc_transposed[44];
        sum_0 += two_column_buffer[3] * ksrc_transposed[45];
        sum_1 += two_column_buffer[4] * ksrc_transposed[45];
        sum_0 += two_column_buffer[4] * ksrc_transposed[46];
        sum_1 += two_column_buffer[5] * ksrc_transposed[46];
        sum_0 += two_column_buffer[5] * ksrc_transposed[47];
        sum_1 += two_column_buffer[6] * ksrc_transposed[47];
        sum_0 += two_column_buffer[6] * ksrc_transposed[48];
        sum_1 += two_column_buffer[7] * ksrc_transposed[48];

        out[(i * output_width + j) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);
        out[(i * output_width + j + 1) * output_depth] = MIN(MAX(sum_1, output_activation_min), output_activation_max);

        two_column_buffer_start += 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        float sum_0 = *bias_data;

        sum_0 += two_column_buffer[0] * ksrc_transposed[0];
        sum_0 += two_column_buffer[1] * ksrc_transposed[1];
        sum_0 += two_column_buffer[2] * ksrc_transposed[2];
        sum_0 += two_column_buffer[3] * ksrc_transposed[3];
        sum_0 += two_column_buffer[4] * ksrc_transposed[4];
        sum_0 += two_column_buffer[5] * ksrc_transposed[5];
        sum_0 += two_column_buffer[6] * ksrc_transposed[6];
        two_column_buffer += input_width * 2 + 5;
        
        sum_0 += two_column_buffer[0] * ksrc_transposed[7];
        sum_0 += two_column_buffer[1] * ksrc_transposed[8];
        sum_0 += two_column_buffer[2] * ksrc_transposed[9];
        sum_0 += two_column_buffer[3] * ksrc_transposed[10];
        sum_0 += two_column_buffer[4] * ksrc_transposed[11];
        sum_0 += two_column_buffer[5] * ksrc_transposed[12];
        sum_0 += two_column_buffer[6] * ksrc_transposed[13];
        two_column_buffer += input_width * 2 + 5;
        
        sum_0 += two_column_buffer[0] * ksrc_transposed[14];
        sum_0 += two_column_buffer[1] * ksrc_transposed[15];
        sum_0 += two_column_buffer[2] * ksrc_transposed[16];
        sum_0 += two_column_buffer[3] * ksrc_transposed[17];
        sum_0 += two_column_buffer[4] * ksrc_transposed[18];
        sum_0 += two_column_buffer[5] * ksrc_transposed[19];
        sum_0 += two_column_buffer[6] * ksrc_transposed[20];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[21];
        sum_0 += two_column_buffer[1] * ksrc_transposed[22];
        sum_0 += two_column_buffer[2] * ksrc_transposed[23];
        sum_0 += two_column_buffer[3] * ksrc_transposed[24];
        sum_0 += two_column_buffer[4] * ksrc_transposed[25];
        sum_0 += two_column_buffer[5] * ksrc_transposed[26];
        sum_0 += two_column_buffer[6] * ksrc_transposed[27];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[28];
        sum_0 += two_column_buffer[1] * ksrc_transposed[29];
        sum_0 += two_column_buffer[2] * ksrc_transposed[30];
        sum_0 += two_column_buffer[3] * ksrc_transposed[31];
        sum_0 += two_column_buffer[4] * ksrc_transposed[32];
        sum_0 += two_column_buffer[5] * ksrc_transposed[33];
        sum_0 += two_column_buffer[6] * ksrc_transposed[34];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[35];
        sum_0 += two_column_buffer[1] * ksrc_transposed[36];
        sum_0 += two_column_buffer[2] * ksrc_transposed[37];
        sum_0 += two_column_buffer[3] * ksrc_transposed[38];
        sum_0 += two_column_buffer[4] * ksrc_transposed[39];
        sum_0 += two_column_buffer[5] * ksrc_transposed[40];
        sum_0 += two_column_buffer[6] * ksrc_transposed[41];
        two_column_buffer += input_width * 2 + 5;

        sum_0 += two_column_buffer[0] * ksrc_transposed[42];
        sum_0 += two_column_buffer[1] * ksrc_transposed[43];
        sum_0 += two_column_buffer[2] * ksrc_transposed[44];
        sum_0 += two_column_buffer[3] * ksrc_transposed[45];
        sum_0 += two_column_buffer[4] * ksrc_transposed[46];
        sum_0 += two_column_buffer[5] * ksrc_transposed[47];
        sum_0 += two_column_buffer[6] * ksrc_transposed[48];

        out[(i * output_width + output_width - 1) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start++;
      }
      /* End of MAC Computation */

      two_column_buffer_start += 6;
    }

    bias_data++;
    input_data++;
    output_data++;
  }
} 
