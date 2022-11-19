/* ----------------------------------------------------------------------
 * Name: transpose_conv_fp_kernelx_stride1_inpad0_outpad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "img2col_element_fp.h"

tinyengine_status_fp transpose_conv_fp_kernelx_stride1_inpad0_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  float* two_column_buffer = im2col_data;
  const int channel_div4 = input_depth >> 2;
  const int filter_width_div4 = filter_width >> 2;
  const int in_row_offset = input_depth * input_width;
  float* out = output_data;
  int i;

  const int num_elements = output_height * output_width * output_depth;
  for (i = 0; i < num_elements; i++) {
    out[i] = 0.0f;
  }

  for (int i_out_y = 0; i_out_y < input_height; i_out_y++) {
    for (int i_out_x = 0; i_out_x < input_width; i_out_x++) {

      /* Im2col for x*x kernel: Start */
      float* col_buffer = &two_column_buffer[0];

      const float* src = &input_data[(i_out_y * input_width + i_out_x) * input_depth];
      float* dst =  &col_buffer[0];
      load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width, filter_width, filter_height);  // Load: x rows, x cols

      two_column_buffer += input_depth * filter_width * filter_height;
      /* Im2col for x*x kernel: End */

      /* Computation is filed for every 2 columns */
      if (two_column_buffer == im2col_data + 2 * input_depth * filter_width * filter_height) {
        float* out2 = out + output_depth;
        const float* bias = bias_data;
        const float* filter_0 = filter_data;

        int num_col_a = filter_width * filter_height * input_depth;
        int row_count = output_depth >> 1;

        /* Compute 4(2x2) output channels each time */
        while (row_count--) {
          const float* input_0 = im2col_data;
          const float* input_1 = input_0 + num_col_a;
          const float* filter_1 = filter_0 + num_col_a;

          float sum_0 = *bias;
          float sum_2 = *bias++;
          float sum_1 = *bias;
          float sum_3 = *bias++;

          int col_count = num_col_a >> 2;

          while (col_count--) {
            sum_0 += *input_0 * *filter_0;
            sum_1 += *input_0++ * *filter_1;
            sum_2 += *input_1 * *filter_0++;
            sum_3 += *input_1++ * *filter_1++;

            sum_0 += *input_0 * *filter_0;
            sum_1 += *input_0++ * *filter_1;
            sum_2 += *input_1 * *filter_0++;
            sum_3 += *input_1++ * *filter_1++;

            sum_0 += *input_0 * *filter_0;
            sum_1 += *input_0++ * *filter_1;
            sum_2 += *input_1 * *filter_0++;
            sum_3 += *input_1++ * *filter_1++;

            sum_0 += *input_0 * *filter_0;
            sum_1 += *input_0++ * *filter_1;
            sum_2 += *input_1 * *filter_0++;
            sum_3 += *input_1++ * *filter_1++;
          }
          col_count = num_col_a & 0x3;
          while (col_count--) {
            sum_0 += *input_0 * *filter_0;
            sum_1 += *input_0++ * *filter_1;
            sum_2 += *input_1 * *filter_0++;
            sum_3 += *input_1++ * *filter_1++;
          }

          *out++ = MIN(MAX(sum_0, output_activation_min), output_activation_max);
          *out++ = MIN(MAX(sum_1, output_activation_min), output_activation_max);
          *out2++ = MIN(MAX(sum_2, output_activation_min), output_activation_max);
          *out2++ = MIN(MAX(sum_3, output_activation_min), output_activation_max);

          filter_0 += num_col_a;
        }

        // if output_depth is odd
        if (output_depth & 0x1) {
          const float* input_0 = im2col_data;
          const float* input_1 = input_0 + num_col_a;

          float sum_0 = *bias;
          float sum_1 = *bias++;

          int col_count = num_col_a >> 2;

          while (col_count--) {
            sum_0 += *input_0++ * *filter_0;
            sum_1 += *input_1++ * *filter_0++;

            sum_0 += *input_0++ * *filter_0;
            sum_1 += *input_1++ * *filter_0++;

            sum_0 += *input_0++ * *filter_0;
            sum_1 += *input_1++ * *filter_0++;

            sum_0 += *input_0++ * *filter_0;
            sum_1 += *input_1++ * *filter_0++;
          }
          col_count = num_col_a & 0x3;
          while (col_count--) {
            sum_0 += *input_0++ * *filter_0;
            sum_1 += *input_1++ * *filter_0++;
          }

          *out++ = MIN(MAX(sum_0, output_activation_min), output_activation_max);
          *out2++ = MIN(MAX(sum_1, output_activation_min), output_activation_max);
        }

        /* counter reset */
        two_column_buffer = im2col_data;
        out += output_depth;
      }
    }
  }

  /* left-over because odd number of output pixels */
  if (two_column_buffer != im2col_data) {
    const float* bias = bias_data;
    const float* filter_0 = filter_data;

    int num_col_a = filter_width * filter_height * input_depth;
    int row_count = output_depth >> 1;

    /* Compute 2 output channels each time */
    while (row_count--) {
      const float* input_0 = im2col_data;
      const float* filter_1 = filter_0 + num_col_a;

      float sum_0 = *bias++;
      float sum_1 = *bias++;

      int col_count = num_col_a >> 2;

      while (col_count--) {
        sum_0 += *input_0 * *filter_0++;
        sum_1 += *input_0++ * *filter_1++;

        sum_0 += *input_0 * *filter_0++;
        sum_1 += *input_0++ * *filter_1++;

        sum_0 += *input_0 * *filter_0++;
        sum_1 += *input_0++ * *filter_1++;

        sum_0 += *input_0 * *filter_0++;
        sum_1 += *input_0++ * *filter_1++;
      }
      col_count = num_col_a & 0x3;
      while (col_count--) {
        sum_0 += *input_0 * *filter_0++;
        sum_1 += *input_0++ * *filter_1++;
      }

      *out++ = MIN(MAX(sum_0, output_activation_min), output_activation_max);
      *out++ = MIN(MAX(sum_1, output_activation_min), output_activation_max);

      filter_0 += num_col_a;
    }

    // if output_depth is odd
    if (output_depth & 0x1) {
      const float* input_0 = im2col_data;
      float sum_0 = *bias;

      int col_count = num_col_a >> 2;

      while (col_count--) {
        sum_0 += *input_0++ * *filter_0++;

        sum_0 += *input_0++ * *filter_0++;

        sum_0 += *input_0++ * *filter_0++;

        sum_0 += *input_0++ * *filter_0++;
      }
      col_count = num_col_a & 0x3;
      while (col_count--) {
        sum_0 += *input_0++ * *filter_0++;
      }

      *out++ = MIN(MAX(sum_0, output_activation_min), output_activation_max);
    }
  }
}
