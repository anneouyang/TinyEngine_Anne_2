/* ----------------------------------------------------------------------
 * Name: depthwise_conv_fp_kernel4_stride1_pad1_dil1.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define DIM_KER_X (8U)
#define DIM_KER_Y (8U)
#define STRIDE (1U)

tinyengine_status_fp depthwise_conv_fp_kernel4_stride1_pad1_in4x4_out3x3_uniweight_1row1col_inplace(const float* input_data, 
                 const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 int8_t* output_weight_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const uint16_t groups,
                 const float* scales, const float learning_rate) {
  (void) input_height;
  (void) input_width;

  const float* filter_data_start = filter_data;
  float* two_column_buffer = im2col_data;
  int i, j, c;

  /* Setup the padding regions for the buffer */
  // Top region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < DIM_KER_Y + 2; i++) {
    *two_column_buffer++ = 0;
  }
  // Middle regions: left and right regions
  for (i = 0; i < DIM_KER_X; i++) {
    *two_column_buffer++ = 0; // left
    two_column_buffer += DIM_KER_Y; // skip middle
    *two_column_buffer++ = 0; // right
  }
  // Bottom region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < DIM_KER_Y + 2; i++) { 
    *two_column_buffer++ = 0;
  }

  /* Brutally use im2col buffer to change filter_data format from HWC -> CHW */
  float* filter_buffer = &im2col_data[(DIM_KER_X + 2) * (DIM_KER_Y + 2)];
  for (c = 0; c < input_depth; c++) {
    filter_data = filter_data_start++;

    for (i = 0; i < DIM_KER_X; i++) {
      for (j = 0; j < DIM_KER_Y; j++) {
        *filter_buffer++ = *filter_data;
        filter_data += input_depth;
      }
    }
  }

  /* Setup the input_data regions for HWC->CHW buffers */
  const float* src;

  // TODO: brutal filter_buffer may overuse the im2col buffer
  float* ksrc = &im2col_data[(DIM_KER_X + 2) * (DIM_KER_Y + 2)];
  //const float* ksrc = filter_data;

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + DIM_KER_Y + 2;
    src = input_data;

    for (i = 0; i < DIM_KER_X; i++) {
      two_column_buffer++;

      for (j = 0; j < DIM_KER_Y; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }

      two_column_buffer++;
    }

    int8_t* out = output_weight_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      // Assume output_width is always a multiple of 3
      for (j = 0; j < output_width - 2; j+=3) {
        two_column_buffer = two_column_buffer_start;

        float sum_0 = 0.0f;
        float sum_1 = 0.0f;
        float sum_2 = 0.0f;

        sum_0 += two_column_buffer[0] * ksrc[0];
        sum_1 += two_column_buffer[1] * ksrc[0];
        sum_2 += two_column_buffer[2] * ksrc[0];
        sum_0 += two_column_buffer[1] * ksrc[1];
        sum_1 += two_column_buffer[2] * ksrc[1];
        sum_2 += two_column_buffer[3] * ksrc[1];
        sum_0 += two_column_buffer[2] * ksrc[2];
        sum_1 += two_column_buffer[3] * ksrc[2];
        sum_2 += two_column_buffer[4] * ksrc[2];
        sum_0 += two_column_buffer[3] * ksrc[3];
        sum_1 += two_column_buffer[4] * ksrc[3];
        sum_2 += two_column_buffer[5] * ksrc[3];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[4];
        sum_1 += two_column_buffer[1] * ksrc[4];
        sum_2 += two_column_buffer[2] * ksrc[4];
        sum_0 += two_column_buffer[1] * ksrc[5];
        sum_1 += two_column_buffer[2] * ksrc[5];
        sum_2 += two_column_buffer[3] * ksrc[5];
        sum_0 += two_column_buffer[2] * ksrc[6];
        sum_1 += two_column_buffer[3] * ksrc[6];
        sum_2 += two_column_buffer[4] * ksrc[6];
        sum_0 += two_column_buffer[3] * ksrc[7];
        sum_1 += two_column_buffer[4] * ksrc[7];
        sum_2 += two_column_buffer[5] * ksrc[7];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[8];
        sum_1 += two_column_buffer[1] * ksrc[8];
        sum_2 += two_column_buffer[2] * ksrc[8];
        sum_0 += two_column_buffer[1] * ksrc[9];
        sum_1 += two_column_buffer[2] * ksrc[9];
        sum_2 += two_column_buffer[3] * ksrc[9];
        sum_0 += two_column_buffer[2] * ksrc[10];
        sum_1 += two_column_buffer[3] * ksrc[10];
        sum_2 += two_column_buffer[4] * ksrc[10];
        sum_0 += two_column_buffer[3] * ksrc[11];
        sum_1 += two_column_buffer[4] * ksrc[11];
        sum_2 += two_column_buffer[5] * ksrc[11];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[12];
        sum_1 += two_column_buffer[1] * ksrc[12];
        sum_2 += two_column_buffer[2] * ksrc[12];
        sum_0 += two_column_buffer[1] * ksrc[13];
        sum_1 += two_column_buffer[2] * ksrc[13];
        sum_2 += two_column_buffer[3] * ksrc[13];
        sum_0 += two_column_buffer[2] * ksrc[14];
        sum_1 += two_column_buffer[3] * ksrc[14];
        sum_2 += two_column_buffer[4] * ksrc[14];
        sum_0 += two_column_buffer[3] * ksrc[15];
        sum_1 += two_column_buffer[4] * ksrc[15];
        sum_2 += two_column_buffer[5] * ksrc[15];

        out[(i * output_width + j) * output_depth] -= MIN(MAX(sum_0, output_activation_min), output_activation_max) * scales[c] * learning_rate;
        out[(i * output_width + j + 1) * output_depth] -= MIN(MAX(sum_1, output_activation_min), output_activation_max) * scales[c] * learning_rate;
        out[(i * output_width + j + 2) * output_depth] -= MIN(MAX(sum_2, output_activation_min), output_activation_max) * scales[c] * learning_rate;
      }

      two_column_buffer_start += DIM_KER_Y + 2;
    }

    ksrc += DIM_KER_X * DIM_KER_Y;
    input_data++;
    output_weight_data++;
  }
  
  /* Return to application */
  return STATE_SUCCESS_fp;
}
