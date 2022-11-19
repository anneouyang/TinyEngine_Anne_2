/* ----------------------------------------------------------------------
 * Name: conv_fp_kernel3_stride1_pad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "img2col_element_fp.h"
#define DIM_KER_X (3U)
#define DIM_KER_Y (3U)

tinyengine_status_fp conv_fp_kernel3_stride1_pad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  float* two_column_buffer = im2col_data;
  const int channel_div4 = (input_depth >> 2);
  int in_row_offset = input_depth * input_width;
  float* out = output_data;

  for (int i_out_y = 0; i_out_y < output_height; i_out_y++) {
    for (int i_out_x = 0; i_out_x < output_width; i_out_x++) {

      /* Im2col for 3x3 kernel: Start */
      const float* src = &input_data[(i_out_y * input_width + i_out_x) * input_depth];
      const float* src2 = src + in_row_offset;
      const float* src3 = src2 + in_row_offset;
      float* dst =  &two_column_buffer[0];
      float* dst2 = &two_column_buffer[input_depth * 3];
      float* dst3 = &two_column_buffer[input_depth * 6];

      load_3row_3col_fp(src, src2, src3, dst, dst2, dst3, channel_div4);

      two_column_buffer += input_depth * DIM_KER_X * DIM_KER_Y;
      /* Im2col for 3x3 kernel: End */

      /* Computation is filed for every 2 columns */
      if (two_column_buffer == im2col_data + 2 * input_depth * DIM_KER_X * DIM_KER_Y) {
        float* out2 = out + output_depth;
        const float* bias = bias_data;
        const float* filter_0 = filter_data;

        int num_col_a = DIM_KER_X * DIM_KER_Y * input_depth;
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

    int num_col_a = DIM_KER_X * DIM_KER_Y * input_depth;
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
