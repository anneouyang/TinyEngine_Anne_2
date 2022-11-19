/* ----------------------------------------------------------------------
 * Name: transpose_conv_fp_kernel5_stride1_inpad2_outpad0.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#include "nnfunctions_fp.h"
#define DIM_KER_X (5U)
#define DIM_KER_Y (5U)
#define STRIDE (1U)

tinyengine_status_fp transpose_conv_fp_kernel5_stride1_inpad2_outpad0(const float* input_data, const uint16_t input_height, const uint16_t input_width, const uint16_t input_depth, 
                 const float* filter_data, const float* bias_data, 
                 float* output_data, const uint16_t final_output_height, const uint16_t final_output_width, const uint16_t output_depth, 
                 const float output_activation_min, const float output_activation_max,
                 float* im2col_data, const uint16_t batches) {
  (void) im2col_data;

  uint16_t output_height = final_output_height + 4;
  uint16_t output_width = final_output_width + 4;
  int i, j ,c;

  const int output_num_elements = output_height * output_width;
  for (i = 0; i < output_depth; i++) {
    for (j = 0; j < output_num_elements; j++) {
      output_data[i + j * output_depth] = bias_data[i];
    }
  }

  for (int i_in_y = 0; i_in_y < input_height; i_in_y++) {
    for (int i_in_x = 0; i_in_x < input_width - 1; i_in_x+=2) {
      float* out = output_data + (i_in_y * output_width + i_in_x) * output_depth;
      float* out2 = out + output_width * output_depth;
      float* out3 = out2 + output_width * output_depth;
      float* out4 = out3 + output_width * output_depth;
      float* out5 = out4 + output_width * output_depth;
      float* out_start = out;
      float* out2_start = out2;
      float* out3_start = out3;
      float* out4_start = out4;
      float* out5_start = out5;

      const float* filter_0 = filter_data;

      int row_count = output_depth >> 1;

      /* Compute 4(2x2) output channels each time */
      while (row_count--) {
        const float* input_0 = input_data + (i_in_y * input_width + i_in_x) * input_depth;
        const float* input_1 = input_0 + input_depth;
        const float* filter_1 = filter_0 + DIM_KER_X * DIM_KER_Y * input_depth;

        int input_depth_div4 = input_depth >> 2;

        /* Assume filter_data (weight) is in the CHW format (instead of HWC format) */
        while (input_depth_div4--) {
          /* input depth 0 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_2filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0, filter_1);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;

          /* input depth 1 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_2filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0, filter_1);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;

          /* input depth 2 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_2filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0, filter_1);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;

          /* input depth 3 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_2filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0, filter_1);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;
        }
        int input_depth_count = input_depth & 0x3;
        while (input_depth_count--) {
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_2filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0, filter_1);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;
        }

        filter_0 += DIM_KER_X * DIM_KER_Y * input_depth;
        out_start += 2;
        out2_start += 2;
        out3_start += 2;
        out4_start += 2;
        out5_start += 2;
      }

      // if output_depth is odd
      if (output_depth & 0x1) {
        const float* input_0 = input_data + (i_in_y * input_width + i_in_x) * input_depth;
        const float* input_1 = input_0 + input_depth;

        int input_depth_div4 = input_depth >> 2;

        /* Assume filter_data (weight) is in the CHW format (instead of HWC format) */
        while (input_depth_div4--) {
          /* input depth 0 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_1filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;

          /* input depth 1 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_1filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;

          /* input depth 2 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_1filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;

          /* input depth 3 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_1filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
        }
        int input_depth_count = input_depth & 0x3;
        while (input_depth_count--) {
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_2input_1filter_1stride_fp(out, out2, out3, out4, out5, output_depth, input_0, input_1, filter_0);
          input_0++; input_1++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
        }
      }
    } // END of: for (int i_in_x = 0; i_in_x < input_width - 1; i_in_x+=2)

    /* left-over because of odd number of input_width */
    if (input_width & 0x1) {
      float* out = output_data + (i_in_y * output_width + (input_width - 1)) * output_depth;
      float* out2 = out + output_width * output_depth;
      float* out3 = out2 + output_width * output_depth;
      float* out4 = out3 + output_width * output_depth;
      float* out5 = out4 + output_width * output_depth;
      float* out_start = out;
      float* out2_start = out2;
      float* out3_start = out3;
      float* out4_start = out4;
      float* out5_start = out5;

      const float* filter_0 = filter_data;

      int row_count = output_depth >> 1;

      /* Compute 4(2x2) output channels each time */
      while (row_count--) {
        const float* input_0 = input_data + (i_in_y * input_width + (input_width - 1)) * input_depth;
        const float* filter_1 = filter_0 + DIM_KER_X * DIM_KER_Y * input_depth;

        int input_depth_div4 = input_depth >> 2;

        /* Assume filter_data (weight) is in the CHW format (instead of HWC format) */
        while (input_depth_div4--) {
          /* input depth 0 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_2filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0, filter_1);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;

          /* input depth 1 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_2filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0, filter_1);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;

          /* input depth 2 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_2filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0, filter_1);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;

          /* input depth 3 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_2filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0, filter_1);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;
        }
        int input_depth_count = input_depth & 0x3;
        while (input_depth_count--) {
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_2filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0, filter_1);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
          filter_1 += DIM_KER_X * DIM_KER_Y;
        }

        filter_0 += DIM_KER_X * DIM_KER_Y * input_depth;
        out_start += 2;
        out2_start += 2;
        out3_start += 2;
        out4_start += 2;
        out5_start += 2;
      }

      // if output_depth is odd
      if (output_depth & 0x1) {
        const float* input_0 = input_data + (i_in_y * input_width + (input_width - 1)) * input_depth;

        int input_depth_div4 = input_depth >> 2;

        /* Assume filter_data (weight) is in the CHW format (instead of HWC format) */
        while (input_depth_div4--) {
          /* input depth 0 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_1filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;

          /* input depth 1 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_1filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;

          /* input depth 2 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_1filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;

          /* input depth 3 */
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_1filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
        }
        int input_depth_count = input_depth & 0x3;
        while (input_depth_count--) {
          out = out_start;
          out2 = out2_start;
          out3 = out3_start;
          out4 = out4_start;
          out5 = out5_start;
          transpose_mac_5row_5col_1input_1filter_xstride_fp(out, out2, out3, out4, out5, output_depth, input_0, filter_0);
          input_0++;
          filter_0 += DIM_KER_X * DIM_KER_Y;
        }
      }
    } // END of: if (input_width & 0x1)
  } // END of: (int i_in_y = 0; i_in_y < input_height; i_in_y++)

  for (c = 0; c < output_depth; c++) {
    for (i = 0; i < final_output_height; i++) {
      for (j = 0; j < final_output_width; j++) {
        output_data[(i * final_output_width + j) * output_depth + c] = output_data[((i+2) * output_width + (j+2)) * output_depth + c];
      }
    }
  }
} 
