/* ----------------------------------------------------------------------
 * Name: transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (7U)
#define DIM_KER_Y (7U)
#define STRIDE (1U)

tinyengine_status_fp transpose_depthwise_conv_fp_kernel7_stride1_inpad3_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  uint16_t output_height = final_output_height + 6;
  uint16_t output_width = final_output_width + 6;
  int i, j, c;

  const int output_num_elements = output_height * output_width;
  for (i = 0; i < output_depth; i++) {
    for (j = 0; j < output_num_elements; j++) {
      output_data[i + j * output_depth] = bias_data[i];
    }
  }

  /* Setup the input_data regions for HWC->CHW buffers */
  float* src;
  const float* ksrc = filter_data;

  for (c = 0; c < input_depth; c++) {
    float* two_column_buffer = im2col_data;
    src = input_data;

    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width; j++) {
        *two_column_buffer++ = *src;
        src += input_depth;
      }
    }

    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < input_height; i++) {
      for (j = 0; j < input_width - 1; j+=2) {
        float* out = output_data + (i * output_width + j) * output_depth + c;
        float* out2 = out + output_width * output_depth;
        float* out3 = out2 + output_width * output_depth;
        float* out4 = out3 + output_width * output_depth;
        float* out5 = out4 + output_width * output_depth;
        float* out6 = out5 + output_width * output_depth;
        float* out7 = out6 + output_width * output_depth;

        const float* input_0 = two_column_buffer_start;
        two_column_buffer_start++;
        const float* input_1 = two_column_buffer_start;

        /* Assume filter_data (weight) is in the CHW format (instead of HWC format) */
        transpose_mac_7row_7col_2input_1filter_1stride_fp(out, out2, out3, out4, out5, out6, out7, output_depth, input_0, input_1, ksrc);

        two_column_buffer_start++;
      }

      /* left-over because odd number of input_width */
      if (input_width & 0x1) {
        float* out = output_data + (i * output_width + (input_width - 1)) * output_depth + c;
        float* out2 = out + output_width * output_depth;
        float* out3 = out2 + output_width * output_depth;
        float* out4 = out3 + output_width * output_depth;
        float* out5 = out4 + output_width * output_depth;
        float* out6 = out5 + output_width * output_depth;
        float* out7 = out6 + output_width * output_depth;

        const float* input_0 = two_column_buffer_start;

        transpose_mac_7row_7col_1input_1filter_xstride_fp(out, out2, out3, out4, out5, out6, out7, output_depth, input_0, ksrc);

        two_column_buffer_start++;
      }
      /* End of MAC Computation */
    }

    ksrc += DIM_KER_X * DIM_KER_Y;
    bias_data++;
    input_data++;
  }

  for (c = 0; c < output_depth; c++) {
    for (i = 0; i < final_output_height; i++) {
      for (j = 0; j < final_output_width; j++) {
        output_data[(i * final_output_width + j) * output_depth + c] = output_data[((i+3) * output_width + (j+3)) * output_depth + c];
      }
    }
  }
} 
