/* ----------------------------------------------------------------------
 * Name: conv_fp_kernelx_stride1_pad1.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "img2col_element_fp.h"

tinyengine_status_fp conv_fp_kernelx_stride1_pad1(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const uint16_t filter_height, const uint16_t filter_width, const float* bias_data, 
                 float* output_data, const uint16_t output_height, const uint16_t output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches, const int pad_value) {
  float* two_column_buffer = im2col_data;
  const int channel_div4 = input_depth >> 2;
  const int filter_width_div4 = filter_width >> 2;
  const int in_row_offset = input_depth * input_width;
  float* out = output_data;

  for (int i_out_y = 0; i_out_y < output_height; i_out_y++) {
    const int16_t base_idx_y = i_out_y - 1;
    for (int i_out_x = 0; i_out_x < output_width; i_out_x++) {
      const int16_t base_idx_x = i_out_x - 1;

      /* Im2col for x*x kernel: Start */
      int block_cnt;
      float* col_buffer = &two_column_buffer[0];

      int ypad_type = 0; //no pad by default
      if (base_idx_y == -1) { //pad the first row
        float* dst = &col_buffer[0];

        // TODO: Could have errors when input_channel % 4 != 0
        int block_cnt = channel_div4 * filter_width_div4;
        while (block_cnt--) {
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //4
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //8
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //12
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //16
        }

        // TODO: Could have errors when input_channel % 4 != 0
        int leftover_block_cnt = channel_div4 * (filter_width & 0x3);
        while (leftover_block_cnt--) {
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; 
        }

        ypad_type = 1;
      }
      else if (base_idx_y + 2 == input_height) { //pad the third row
        float* dst = &col_buffer[input_depth * (filter_height - 1) * filter_width];

        // TODO: Could have errors when input_channel % 4 != 0
        int block_cnt = channel_div4 * filter_width_div4;
        while (block_cnt--) {
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //4
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //8
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //12
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; //16
        }

        // TODO: Could have errors when input_channel % 4 != 0
        int leftover_block_cnt = channel_div4 * (filter_width & 0x3);
        while (leftover_block_cnt--) {
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value;
          *dst++ = pad_value; 
        }

        ypad_type = 2;
      }

      if (ypad_type == 0) { // fill all rows
        if (base_idx_x == -1) {
          /* use pad for the first 1 col */
          float* dst =  &col_buffer[0];

          pad_xrow_1col_fp(dst, input_depth, filter_width, filter_height, pad_value); // Pad: x rows, 1 col

          /* load input to the last 2 col */
          const float* src = &input_data[base_idx_y * input_width * input_depth];
          dst += input_depth;

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width - 1, filter_width, filter_height); // Load: x rows, (x-1) cols
        }
        else if (base_idx_x + 2 == input_width) {
          /* load input to the first 2 col */
          const float* src = &input_data[(base_idx_y * input_width + base_idx_x) * input_depth];
          float* dst =  &col_buffer[0];

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width - 1, filter_width, filter_height); // Load: x rows, (x-1) cols

          /* use pad for the last 1 col */
          dst += input_depth * (filter_width - 1);

          pad_xrow_1col_fp(dst, input_depth, filter_width, filter_height, pad_value); // Pad: x rows, 1 col
        }
        else {
          /* load input to all the 3 col */
          const float* src = &input_data[(base_idx_y * input_width + base_idx_x) * input_depth];
          float* dst =  &col_buffer[0];

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width, filter_width, filter_height);  // Load: x rows, x cols
        }
      }
      else if (ypad_type == 1) { //fill the last two rows
        if (base_idx_x == -1) {
          /* use pad for the first 1 col */
          float* dst = &col_buffer[input_depth * filter_width];

          pad_xrow_1col_fp(dst, input_depth, filter_width, filter_height - 1, pad_value); // Pad: (x-1) rows, 1 col

          /* load input to the last 2 col */
          const float* src = input_data;
          dst += input_depth;

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width - 1, filter_width, filter_height - 1); // Load: (x-1) rows, (x-1) cols
        }
        else if (base_idx_x + 2 == input_width) {
          /* load input to the first 2 col */
          const float* src = &input_data[base_idx_x * input_depth];
          float* dst = &col_buffer[input_depth * filter_width];

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width - 1, filter_width, filter_height - 1); // Load: (x-1) rows, (x-1) cols

          /* use pad for the last 1 col */
          dst += input_depth * (filter_width - 1);

          pad_xrow_1col_fp(dst, input_depth, filter_width, filter_height - 1, pad_value); // Pad: (x-1) rows, 1 col
        }
        else {
          /* load the input to all the 3 col */
          const float* src = &input_data[base_idx_x * input_depth];
          float* dst = &col_buffer[input_depth * filter_width];

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width, filter_width, filter_height - 1); // Load: (x-1) rows, x cols
        }
      }
      else { // fill the first two rows. (ypad_type == 2)
        if (base_idx_x == -1) {
          /* use pad for the first 1 col */
          float* dst = &col_buffer[0];

          pad_xrow_1col_fp(dst, input_depth, filter_width, filter_height - 1, pad_value); // Pad: (x-1) rows, 1 col

          /* load input to the last 2 col */
          const float* src = &input_data[base_idx_y * input_width * input_depth];
          dst += input_depth;
          
          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width - 1, filter_width, filter_height - 1); // Load: (x-1) rows, (x-1) cols
        }
        else if (base_idx_x + 2 == input_width) {
          /* load input to the first 2 col */
          const float* src = &input_data[(base_idx_y * input_width + base_idx_x) * input_depth];
          float* dst = &col_buffer[0];

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width - 1, filter_width, filter_height - 1); // Load: (x-1) rows, (x-1) cols

          /* use pad for the last 1 col */
          dst += input_depth * (filter_width - 1);

          pad_xrow_1col_fp(dst, input_depth, filter_width, filter_height - 1, pad_value); // Pad: (x-1) rows, 1 col
        }
        else {
          /* load input to all the 3 col */
          const float* src = &input_data[(base_idx_y * input_width + base_idx_x) * input_depth];
          float* dst = &col_buffer[0];

          load_xrow_xcol_fp(src, dst, input_width, input_depth, filter_width, filter_width, filter_height - 1); // Load: (x-1) rows, x cols
        }
      }

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
