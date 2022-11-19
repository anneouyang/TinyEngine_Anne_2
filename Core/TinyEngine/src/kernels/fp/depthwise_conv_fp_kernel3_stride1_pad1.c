/* ----------------------------------------------------------------------
 * Name: depthwise_conv_fp_kernel3_stride1_pad1.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define DIM_KER_X (3U)
#define DIM_KER_Y (3U)
#define STRIDE (1U)

tinyengine_status_fp depthwise_conv_fp_kernel3_stride1_pad1 (float* input_output_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  int i;

  /* Setup the padding regions for the buffer */
  // Top region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < input_width + 2; i++) {
    *two_column_buffer++ = pad_value;
  }
  // Middle regions: left and right regions
  for (i = 0; i < input_height; i++) {
    *two_column_buffer++ = pad_value; // left
    two_column_buffer += input_width; // skip middle
    *two_column_buffer++ = pad_value; // right
  }
  // Bottom region: 8bit x (input_x + pad_w * 2) x pad_h: unroll by pad_value
  for (i = 0; i < input_width + 2; i++) { 
    *two_column_buffer++ = pad_value;
  }

  /* Setup the input_output_data regions for HWC->CHW buffers */
  const float* src;
  const float* ksrc = filter_data;
  int j, c;

  for (c = 0; c < input_depth; c++) {
    two_column_buffer = im2col_data + input_width + 2;
    src = input_output_data;

    for (i = 0; i < input_height; i++) {
      two_column_buffer++;

      for (j = 0; j < input_width; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }

      two_column_buffer++;
    }

    float* inplace_output = input_output_data;
    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 1; j+=2) {
        two_column_buffer = two_column_buffer_start;

        float sum_0 = *bias_data;
        float sum_1 = *bias_data;

        sum_0 += two_column_buffer[0] * ksrc[0];
        sum_1 += two_column_buffer[1] * ksrc[0];
        sum_0 += two_column_buffer[1] * ksrc[1];
        sum_1 += two_column_buffer[2] * ksrc[1];
        sum_0 += two_column_buffer[2] * ksrc[2];
        sum_1 += two_column_buffer[3] * ksrc[2];
        two_column_buffer += input_width + 2;
        
        sum_0 += two_column_buffer[0] * ksrc[3];
        sum_1 += two_column_buffer[1] * ksrc[3];
        sum_0 += two_column_buffer[1] * ksrc[4];
        sum_1 += two_column_buffer[2] * ksrc[4];
        sum_0 += two_column_buffer[2] * ksrc[5];
        sum_1 += two_column_buffer[3] * ksrc[5];
        two_column_buffer += input_width + 2;
        
        sum_0 += two_column_buffer[0] * ksrc[6];
        sum_1 += two_column_buffer[1] * ksrc[6];
        sum_0 += two_column_buffer[1] * ksrc[7];
        sum_1 += two_column_buffer[2] * ksrc[7];
        sum_0 += two_column_buffer[2] * ksrc[8];
        sum_1 += two_column_buffer[3] * ksrc[8];

        inplace_output[(i * output_width + j) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);
        inplace_output[(i * output_width + j + 1) * output_depth] = MIN(MAX(sum_1, output_activation_min), output_activation_max);

        two_column_buffer_start += STRIDE * 2;
      }

      /* left-over because odd number of output pixels */
      if (output_width & 0x1) {
        two_column_buffer = two_column_buffer_start;

        float sum_0 = *bias_data;

        sum_0 += two_column_buffer[0] * ksrc[0];
        sum_0 += two_column_buffer[1] * ksrc[1];
        sum_0 += two_column_buffer[2] * ksrc[2];
        two_column_buffer += input_width + 2;
        
        sum_0 += two_column_buffer[0] * ksrc[3];
        sum_0 += two_column_buffer[1] * ksrc[4];
        sum_0 += two_column_buffer[2] * ksrc[5];
        two_column_buffer += input_width + 2;
        
        sum_0 += two_column_buffer[0] * ksrc[6];
        sum_0 += two_column_buffer[1] * ksrc[7];
        sum_0 += two_column_buffer[2] * ksrc[8];

        inplace_output[(i * output_width + output_width - 1) * output_depth] = MIN(MAX(sum_0, output_activation_min), output_activation_max);

        two_column_buffer_start += STRIDE;
      }
      /* End of MAC Computation */

      two_column_buffer_start += STRIDE * 2;
    }

    ksrc += DIM_KER_X * DIM_KER_Y;
    bias_data++;
    input_output_data++;
  }
}
