/* ----------------------------------------------------------------------
 * Name: transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_inw3_revised.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (7U)
#define DIM_KER_Y (7U)
#define STRIDE (1U)
#define IN_PAD (3U)
#define OUT_PAD (0U)

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0_inw3_revised(float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
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
  const float* src;
  const float* ksrc = filter_data;
  float ksrc_transposed[49];

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + (input_width + 6) * 3;
    src = input_output_data;

    // Place input data into two_column_buffer
    for (i = 0; i < input_height; i++) {
      two_column_buffer += 3;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }

      two_column_buffer += 3;
    }

    // Transpose filter data
    for (i = 0; i < DIM_KER_Y * DIM_KER_X; i++) {
      ksrc_transposed[48 - i] = *ksrc++;
    }

    float* inplace_output = input_output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      two_column_buffer = two_column_buffer_start;

      /*
      float sum_0 = *bias_data;
      float sum_1 = *bias_data;
      float sum_2 = *bias_data;
      */
      float sum_0 = 0.0f;
      float sum_1 = 0.0f;
      float sum_2 = 0.0f;

      sum_0 += two_column_buffer[0] * ksrc_transposed[0];
      sum_1 += two_column_buffer[1] * ksrc_transposed[0];
      sum_2 += two_column_buffer[2] * ksrc_transposed[0];
      sum_0 += two_column_buffer[1] * ksrc_transposed[1];
      sum_1 += two_column_buffer[2] * ksrc_transposed[1];
      sum_2 += two_column_buffer[3] * ksrc_transposed[1];
      sum_0 += two_column_buffer[2] * ksrc_transposed[2];
      sum_1 += two_column_buffer[3] * ksrc_transposed[2];
      sum_2 += two_column_buffer[4] * ksrc_transposed[2];
      sum_0 += two_column_buffer[3] * ksrc_transposed[3];
      sum_1 += two_column_buffer[4] * ksrc_transposed[3];
      sum_2 += two_column_buffer[5] * ksrc_transposed[3];
      sum_0 += two_column_buffer[4] * ksrc_transposed[4];
      sum_1 += two_column_buffer[5] * ksrc_transposed[4];
      sum_2 += two_column_buffer[6] * ksrc_transposed[4];
      sum_0 += two_column_buffer[5] * ksrc_transposed[5];
      sum_1 += two_column_buffer[6] * ksrc_transposed[5];
      sum_2 += two_column_buffer[7] * ksrc_transposed[5];
      sum_0 += two_column_buffer[6] * ksrc_transposed[6];
      sum_1 += two_column_buffer[7] * ksrc_transposed[6];
      sum_2 += two_column_buffer[8] * ksrc_transposed[6];
      two_column_buffer += input_width + 6;
      
      sum_0 += two_column_buffer[0] * ksrc_transposed[7];
      sum_1 += two_column_buffer[1] * ksrc_transposed[7];
      sum_2 += two_column_buffer[2] * ksrc_transposed[7];
      sum_0 += two_column_buffer[1] * ksrc_transposed[8];
      sum_1 += two_column_buffer[2] * ksrc_transposed[8];
      sum_2 += two_column_buffer[3] * ksrc_transposed[8];
      sum_0 += two_column_buffer[2] * ksrc_transposed[9];
      sum_1 += two_column_buffer[3] * ksrc_transposed[9];
      sum_2 += two_column_buffer[4] * ksrc_transposed[9];
      sum_0 += two_column_buffer[3] * ksrc_transposed[10];
      sum_1 += two_column_buffer[4] * ksrc_transposed[10];
      sum_2 += two_column_buffer[5] * ksrc_transposed[10];
      sum_0 += two_column_buffer[4] * ksrc_transposed[11];
      sum_1 += two_column_buffer[5] * ksrc_transposed[11];
      sum_2 += two_column_buffer[6] * ksrc_transposed[11];
      sum_0 += two_column_buffer[5] * ksrc_transposed[12];
      sum_1 += two_column_buffer[6] * ksrc_transposed[12];
      sum_2 += two_column_buffer[7] * ksrc_transposed[12];
      sum_0 += two_column_buffer[6] * ksrc_transposed[13];
      sum_1 += two_column_buffer[7] * ksrc_transposed[13];
      sum_2 += two_column_buffer[8] * ksrc_transposed[13];
      two_column_buffer += input_width + 6;
      
      sum_0 += two_column_buffer[0] * ksrc_transposed[14];
      sum_1 += two_column_buffer[1] * ksrc_transposed[14];
      sum_2 += two_column_buffer[2] * ksrc_transposed[14];
      sum_0 += two_column_buffer[1] * ksrc_transposed[15];
      sum_1 += two_column_buffer[2] * ksrc_transposed[15];
      sum_2 += two_column_buffer[3] * ksrc_transposed[15];
      sum_0 += two_column_buffer[2] * ksrc_transposed[16];
      sum_1 += two_column_buffer[3] * ksrc_transposed[16];
      sum_2 += two_column_buffer[4] * ksrc_transposed[16];
      sum_0 += two_column_buffer[3] * ksrc_transposed[17];
      sum_1 += two_column_buffer[4] * ksrc_transposed[17];
      sum_2 += two_column_buffer[5] * ksrc_transposed[17];
      sum_0 += two_column_buffer[4] * ksrc_transposed[18];
      sum_1 += two_column_buffer[5] * ksrc_transposed[18];
      sum_2 += two_column_buffer[6] * ksrc_transposed[18];
      sum_0 += two_column_buffer[5] * ksrc_transposed[19];
      sum_1 += two_column_buffer[6] * ksrc_transposed[19];
      sum_2 += two_column_buffer[7] * ksrc_transposed[19];
      sum_0 += two_column_buffer[6] * ksrc_transposed[20];
      sum_1 += two_column_buffer[7] * ksrc_transposed[20];
      sum_2 += two_column_buffer[8] * ksrc_transposed[20];
      two_column_buffer += input_width + 6;

      sum_0 += two_column_buffer[0] * ksrc_transposed[21];
      sum_1 += two_column_buffer[1] * ksrc_transposed[21];
      sum_2 += two_column_buffer[2] * ksrc_transposed[21];
      sum_0 += two_column_buffer[1] * ksrc_transposed[22];
      sum_1 += two_column_buffer[2] * ksrc_transposed[22];
      sum_2 += two_column_buffer[3] * ksrc_transposed[22];
      sum_0 += two_column_buffer[2] * ksrc_transposed[23];
      sum_1 += two_column_buffer[3] * ksrc_transposed[23];
      sum_2 += two_column_buffer[4] * ksrc_transposed[23];
      sum_0 += two_column_buffer[3] * ksrc_transposed[24];
      sum_1 += two_column_buffer[4] * ksrc_transposed[24];
      sum_2 += two_column_buffer[5] * ksrc_transposed[24];
      sum_0 += two_column_buffer[4] * ksrc_transposed[25];
      sum_1 += two_column_buffer[5] * ksrc_transposed[25];
      sum_2 += two_column_buffer[6] * ksrc_transposed[25];
      sum_0 += two_column_buffer[5] * ksrc_transposed[26];
      sum_1 += two_column_buffer[6] * ksrc_transposed[26];
      sum_2 += two_column_buffer[7] * ksrc_transposed[26];
      sum_0 += two_column_buffer[6] * ksrc_transposed[27];
      sum_1 += two_column_buffer[7] * ksrc_transposed[27];
      sum_2 += two_column_buffer[8] * ksrc_transposed[27];
      two_column_buffer += input_width + 6;

      sum_0 += two_column_buffer[0] * ksrc_transposed[28];
      sum_1 += two_column_buffer[1] * ksrc_transposed[28];
      sum_2 += two_column_buffer[2] * ksrc_transposed[28];
      sum_0 += two_column_buffer[1] * ksrc_transposed[29];
      sum_1 += two_column_buffer[2] * ksrc_transposed[29];
      sum_2 += two_column_buffer[3] * ksrc_transposed[29];
      sum_0 += two_column_buffer[2] * ksrc_transposed[30];
      sum_1 += two_column_buffer[3] * ksrc_transposed[30];
      sum_2 += two_column_buffer[4] * ksrc_transposed[30];
      sum_0 += two_column_buffer[3] * ksrc_transposed[31];
      sum_1 += two_column_buffer[4] * ksrc_transposed[31];
      sum_2 += two_column_buffer[5] * ksrc_transposed[31];
      sum_0 += two_column_buffer[4] * ksrc_transposed[32];
      sum_1 += two_column_buffer[5] * ksrc_transposed[32];
      sum_2 += two_column_buffer[6] * ksrc_transposed[32];
      sum_0 += two_column_buffer[5] * ksrc_transposed[33];
      sum_1 += two_column_buffer[6] * ksrc_transposed[33];
      sum_2 += two_column_buffer[7] * ksrc_transposed[33];
      sum_0 += two_column_buffer[6] * ksrc_transposed[34];
      sum_1 += two_column_buffer[7] * ksrc_transposed[34];
      sum_2 += two_column_buffer[8] * ksrc_transposed[34];
      two_column_buffer += input_width + 6;

      sum_0 += two_column_buffer[0] * ksrc_transposed[35];
      sum_1 += two_column_buffer[1] * ksrc_transposed[35];
      sum_2 += two_column_buffer[2] * ksrc_transposed[35];
      sum_0 += two_column_buffer[1] * ksrc_transposed[36];
      sum_1 += two_column_buffer[2] * ksrc_transposed[36];
      sum_2 += two_column_buffer[3] * ksrc_transposed[36];
      sum_0 += two_column_buffer[2] * ksrc_transposed[37];
      sum_1 += two_column_buffer[3] * ksrc_transposed[37];
      sum_2 += two_column_buffer[4] * ksrc_transposed[37];
      sum_0 += two_column_buffer[3] * ksrc_transposed[38];
      sum_1 += two_column_buffer[4] * ksrc_transposed[38];
      sum_2 += two_column_buffer[5] * ksrc_transposed[38];
      sum_0 += two_column_buffer[4] * ksrc_transposed[39];
      sum_1 += two_column_buffer[5] * ksrc_transposed[39];
      sum_2 += two_column_buffer[6] * ksrc_transposed[39];
      sum_0 += two_column_buffer[5] * ksrc_transposed[40];
      sum_1 += two_column_buffer[6] * ksrc_transposed[40];
      sum_2 += two_column_buffer[7] * ksrc_transposed[40];
      sum_0 += two_column_buffer[6] * ksrc_transposed[41];
      sum_1 += two_column_buffer[7] * ksrc_transposed[41];
      sum_2 += two_column_buffer[8] * ksrc_transposed[41];
      two_column_buffer += input_width + 6;

      sum_0 += two_column_buffer[0] * ksrc_transposed[42];
      sum_1 += two_column_buffer[1] * ksrc_transposed[42];
      sum_2 += two_column_buffer[2] * ksrc_transposed[42];
      sum_0 += two_column_buffer[1] * ksrc_transposed[43];
      sum_1 += two_column_buffer[2] * ksrc_transposed[43];
      sum_2 += two_column_buffer[3] * ksrc_transposed[43];
      sum_0 += two_column_buffer[2] * ksrc_transposed[44];
      sum_1 += two_column_buffer[3] * ksrc_transposed[44];
      sum_2 += two_column_buffer[4] * ksrc_transposed[44];
      sum_0 += two_column_buffer[3] * ksrc_transposed[45];
      sum_1 += two_column_buffer[4] * ksrc_transposed[45];
      sum_2 += two_column_buffer[5] * ksrc_transposed[45];
      sum_0 += two_column_buffer[4] * ksrc_transposed[46];
      sum_1 += two_column_buffer[5] * ksrc_transposed[46];
      sum_2 += two_column_buffer[6] * ksrc_transposed[46];
      sum_0 += two_column_buffer[5] * ksrc_transposed[47];
      sum_1 += two_column_buffer[6] * ksrc_transposed[47];
      sum_2 += two_column_buffer[7] * ksrc_transposed[47];
      sum_0 += two_column_buffer[6] * ksrc_transposed[48];
      sum_1 += two_column_buffer[7] * ksrc_transposed[48];
      sum_2 += two_column_buffer[8] * ksrc_transposed[48];

      inplace_output[i * output_width* output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);
      inplace_output[(i * output_width + 1) * output_depth] = MIN(MAX(sum_1, output_activation_min), output_activation_max);
      inplace_output[(i * output_width + 2) * output_depth] = MIN(MAX(sum_2, output_activation_min), output_activation_max);

      /* End of MAC Computation */
      two_column_buffer_start += 9;
    }

    bias_data++;
    input_output_data++;
  }
} 
