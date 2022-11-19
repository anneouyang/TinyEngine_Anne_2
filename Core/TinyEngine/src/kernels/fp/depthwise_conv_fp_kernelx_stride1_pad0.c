/* ----------------------------------------------------------------------
 * Name: depthwise_conv_fp_kernelx_stride1_pad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define STRIDE (1U)

tinyengine_status_fp depthwise_conv_fp_kernelx_stride1_pad0 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
                 const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  float* two_column_buffer = im2col_data;

  /* Setup the input_output_data regions for HWC->CHW buffers */
  const float* src;
  const float* ksrc = filter_data;
  int i, j, c;

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data;
    src = input_output_data;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }
    }

    float* inplace_output = input_output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      // TODO: Could have errors when output_width < 4
      for (j = 0; j < output_width - 3; j+=4) {
        two_column_buffer = two_column_buffer_start;
        ksrc = filter_data;

        float sum_0 = *bias_data;
        float sum_1 = *bias_data;
        float sum_2 = *bias_data;
        float sum_3 = *bias_data;

        int m, n;
        for (m = 0; m < filter_height; m++) {
          int cnt_filter_width = filter_width / 8;
          while (cnt_filter_width--) {
            sum_0 += two_column_buffer[0] * ksrc[0];
            sum_1 += two_column_buffer[1] * ksrc[0];
            sum_2 += two_column_buffer[2] * ksrc[0];
            sum_3 += two_column_buffer[3] * ksrc[0];
            sum_0 += two_column_buffer[1] * ksrc[1];
            sum_1 += two_column_buffer[2] * ksrc[1];
            sum_2 += two_column_buffer[3] * ksrc[1];
            sum_3 += two_column_buffer[4] * ksrc[1];
            sum_0 += two_column_buffer[2] * ksrc[2];
            sum_1 += two_column_buffer[3] * ksrc[2];
            sum_2 += two_column_buffer[4] * ksrc[2];
            sum_3 += two_column_buffer[5] * ksrc[2];
            sum_0 += two_column_buffer[3] * ksrc[3];
            sum_1 += two_column_buffer[4] * ksrc[3];
            sum_2 += two_column_buffer[5] * ksrc[3];
            sum_3 += two_column_buffer[6] * ksrc[3];
            sum_0 += two_column_buffer[4] * ksrc[4];
            sum_1 += two_column_buffer[5] * ksrc[4];
            sum_2 += two_column_buffer[6] * ksrc[4];
            sum_3 += two_column_buffer[7] * ksrc[4];
            sum_0 += two_column_buffer[5] * ksrc[5];
            sum_1 += two_column_buffer[6] * ksrc[5];
            sum_2 += two_column_buffer[7] * ksrc[5];
            sum_3 += two_column_buffer[8] * ksrc[5];
            sum_0 += two_column_buffer[6] * ksrc[6];
            sum_1 += two_column_buffer[7] * ksrc[6];
            sum_2 += two_column_buffer[8] * ksrc[6];
            sum_3 += two_column_buffer[9] * ksrc[6];
            sum_0 += two_column_buffer[7] * ksrc[7];
            sum_1 += two_column_buffer[8] * ksrc[7];
            sum_2 += two_column_buffer[9] * ksrc[7];
            sum_3 += two_column_buffer[10] * ksrc[7];

            two_column_buffer += 8;
            ksrc += 8;
          }

          int leftover_filter_width = filter_width % 8;
          while (leftover_filter_width--) {
            sum_0 += two_column_buffer[0] * ksrc[0];
            sum_1 += two_column_buffer[1] * ksrc[0];
            sum_2 += two_column_buffer[2] * ksrc[0];
            sum_3 += two_column_buffer[3] * ksrc[0];

            two_column_buffer++;
            ksrc++;
          }

          two_column_buffer = two_column_buffer_start + (m + 1) * input_width;
        }

        inplace_output[(i * output_width + j) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);
        inplace_output[(i * output_width + j + 1) * output_depth] = MIN(MAX(sum_1, output_activation_min), output_activation_max);
        inplace_output[(i * output_width + j + 2) * output_depth] = MIN(MAX(sum_2, output_activation_min), output_activation_max);
        inplace_output[(i * output_width + j + 3) * output_depth] = MIN(MAX(sum_3, output_activation_min), output_activation_max);

        two_column_buffer_start += STRIDE * 4;
      }

      /* left-over because odd number of output pixels */
      int leftover_output_width = output_width & 0x3;
      while (leftover_output_width) {
        two_column_buffer = two_column_buffer_start;
        ksrc = filter_data;

        float sum_0 = *bias_data;

        int m, n;
        for (m = 0; m < filter_height; m++) {
          int cnt_filter_width = filter_width / 8;
          while (cnt_filter_width--) {
            sum_0 += two_column_buffer[0] * ksrc[0];
            sum_0 += two_column_buffer[1] * ksrc[1];
            sum_0 += two_column_buffer[2] * ksrc[2];
            sum_0 += two_column_buffer[3] * ksrc[3];
            sum_0 += two_column_buffer[4] * ksrc[4];
            sum_0 += two_column_buffer[5] * ksrc[5];
            sum_0 += two_column_buffer[6] * ksrc[6];
            sum_0 += two_column_buffer[7] * ksrc[7];

            two_column_buffer += 8;
            ksrc += 8;
          }

          int leftover_filter_width = filter_width % 8;
          while (leftover_filter_width--) {
            sum_0 += two_column_buffer[0] * ksrc[0];

            two_column_buffer++;
            ksrc++;
          }

          two_column_buffer = two_column_buffer_start + (m + 1) * input_width;
        }

        inplace_output[(i * output_width + (output_width - leftover_output_width)) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start += STRIDE;
        leftover_output_width--;
      }
      /* End of MAC Computation */

      two_column_buffer_start += STRIDE * (filter_width - 1);
    }

    filter_data += filter_width * filter_height;
    bias_data++;
    input_output_data++;
  }
}
