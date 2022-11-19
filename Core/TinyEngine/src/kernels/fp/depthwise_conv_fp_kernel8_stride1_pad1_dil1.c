/* ----------------------------------------------------------------------
 * Name: depthwise_conv_fp_kernel8_stride1_pad1_dil1.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define DIM_KER_X (8U)
#define DIM_KER_Y (8U)
#define STRIDE (1U)

tinyengine_status_fp depthwise_conv_fp_kernel8_stride1_pad1_dil1_in8x8_out3x3_uniweight_1row1col_inplace(const float* input_data, 
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

    float* two_column_buffer_start = im2col_data;

    /* MAC Computation */
    for (i = 0; i < output_height; i++) {
      for (j = 0; j < output_width - 2; j+=3) {
        two_column_buffer = two_column_buffer_start;

        float sum_0 = 0;
        float sum_1 = 0;
        float sum_2 = 0;

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
        sum_0 += two_column_buffer[4] * ksrc[4];
        sum_1 += two_column_buffer[5] * ksrc[4];
        sum_2 += two_column_buffer[6] * ksrc[4];
        sum_0 += two_column_buffer[5] * ksrc[5];
        sum_1 += two_column_buffer[6] * ksrc[5];
        sum_2 += two_column_buffer[7] * ksrc[5];
        sum_0 += two_column_buffer[6] * ksrc[6];
        sum_1 += two_column_buffer[7] * ksrc[6];
        sum_2 += two_column_buffer[8] * ksrc[6];
        sum_0 += two_column_buffer[7] * ksrc[7];
        sum_1 += two_column_buffer[8] * ksrc[7];
        sum_2 += two_column_buffer[9] * ksrc[7];
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
        sum_0 += two_column_buffer[4] * ksrc[12];
        sum_1 += two_column_buffer[5] * ksrc[12];
        sum_2 += two_column_buffer[6] * ksrc[12];
        sum_0 += two_column_buffer[5] * ksrc[13];
        sum_1 += two_column_buffer[6] * ksrc[13];
        sum_2 += two_column_buffer[7] * ksrc[13];
        sum_0 += two_column_buffer[6] * ksrc[14];
        sum_1 += two_column_buffer[7] * ksrc[14];
        sum_2 += two_column_buffer[8] * ksrc[14];
        sum_0 += two_column_buffer[7] * ksrc[15];
        sum_1 += two_column_buffer[8] * ksrc[15];
        sum_2 += two_column_buffer[9] * ksrc[15];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[16];
        sum_1 += two_column_buffer[1] * ksrc[16];
        sum_2 += two_column_buffer[2] * ksrc[16];
        sum_0 += two_column_buffer[1] * ksrc[17];
        sum_1 += two_column_buffer[2] * ksrc[17];
        sum_2 += two_column_buffer[3] * ksrc[17];
        sum_0 += two_column_buffer[2] * ksrc[18];
        sum_1 += two_column_buffer[3] * ksrc[18];
        sum_2 += two_column_buffer[4] * ksrc[18];
        sum_0 += two_column_buffer[3] * ksrc[19];
        sum_1 += two_column_buffer[4] * ksrc[19];
        sum_2 += two_column_buffer[5] * ksrc[19];
        sum_0 += two_column_buffer[4] * ksrc[20];
        sum_1 += two_column_buffer[5] * ksrc[20];
        sum_2 += two_column_buffer[6] * ksrc[20];
        sum_0 += two_column_buffer[5] * ksrc[21];
        sum_1 += two_column_buffer[6] * ksrc[21];
        sum_2 += two_column_buffer[7] * ksrc[21];
        sum_0 += two_column_buffer[6] * ksrc[22];
        sum_1 += two_column_buffer[7] * ksrc[22];
        sum_2 += two_column_buffer[8] * ksrc[22];
        sum_0 += two_column_buffer[7] * ksrc[23];
        sum_1 += two_column_buffer[8] * ksrc[23];
        sum_2 += two_column_buffer[9] * ksrc[23];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[24];
        sum_1 += two_column_buffer[1] * ksrc[24];
        sum_2 += two_column_buffer[2] * ksrc[24];
        sum_0 += two_column_buffer[1] * ksrc[25];
        sum_1 += two_column_buffer[2] * ksrc[25];
        sum_2 += two_column_buffer[3] * ksrc[25];
        sum_0 += two_column_buffer[2] * ksrc[26];
        sum_1 += two_column_buffer[3] * ksrc[26];
        sum_2 += two_column_buffer[4] * ksrc[26];
        sum_0 += two_column_buffer[3] * ksrc[27];
        sum_1 += two_column_buffer[4] * ksrc[27];
        sum_2 += two_column_buffer[5] * ksrc[27];
        sum_0 += two_column_buffer[4] * ksrc[28];
        sum_1 += two_column_buffer[5] * ksrc[28];
        sum_2 += two_column_buffer[6] * ksrc[28];
        sum_0 += two_column_buffer[5] * ksrc[29];
        sum_1 += two_column_buffer[6] * ksrc[29];
        sum_2 += two_column_buffer[7] * ksrc[29];
        sum_0 += two_column_buffer[6] * ksrc[30];
        sum_1 += two_column_buffer[7] * ksrc[30];
        sum_2 += two_column_buffer[8] * ksrc[30];
        sum_0 += two_column_buffer[7] * ksrc[31];
        sum_1 += two_column_buffer[8] * ksrc[31];
        sum_2 += two_column_buffer[9] * ksrc[31];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[32];
        sum_1 += two_column_buffer[1] * ksrc[32];
        sum_2 += two_column_buffer[2] * ksrc[32];
        sum_0 += two_column_buffer[1] * ksrc[33];
        sum_1 += two_column_buffer[2] * ksrc[33];
        sum_2 += two_column_buffer[3] * ksrc[33];
        sum_0 += two_column_buffer[2] * ksrc[34];
        sum_1 += two_column_buffer[3] * ksrc[34];
        sum_2 += two_column_buffer[4] * ksrc[34];
        sum_0 += two_column_buffer[3] * ksrc[35];
        sum_1 += two_column_buffer[4] * ksrc[35];
        sum_2 += two_column_buffer[5] * ksrc[35];
        sum_0 += two_column_buffer[4] * ksrc[36];
        sum_1 += two_column_buffer[5] * ksrc[36];
        sum_2 += two_column_buffer[6] * ksrc[36];
        sum_0 += two_column_buffer[5] * ksrc[37];
        sum_1 += two_column_buffer[6] * ksrc[37];
        sum_2 += two_column_buffer[7] * ksrc[37];
        sum_0 += two_column_buffer[6] * ksrc[38];
        sum_1 += two_column_buffer[7] * ksrc[38];
        sum_2 += two_column_buffer[8] * ksrc[38];
        sum_0 += two_column_buffer[7] * ksrc[39];
        sum_1 += two_column_buffer[8] * ksrc[39];
        sum_2 += two_column_buffer[9] * ksrc[39];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[40];
        sum_1 += two_column_buffer[1] * ksrc[40];
        sum_2 += two_column_buffer[2] * ksrc[40];
        sum_0 += two_column_buffer[1] * ksrc[41];
        sum_1 += two_column_buffer[2] * ksrc[41];
        sum_2 += two_column_buffer[3] * ksrc[41];
        sum_0 += two_column_buffer[2] * ksrc[42];
        sum_1 += two_column_buffer[3] * ksrc[42];
        sum_2 += two_column_buffer[4] * ksrc[42];
        sum_0 += two_column_buffer[3] * ksrc[43];
        sum_1 += two_column_buffer[4] * ksrc[43];
        sum_2 += two_column_buffer[5] * ksrc[43];
        sum_0 += two_column_buffer[4] * ksrc[44];
        sum_1 += two_column_buffer[5] * ksrc[44];
        sum_2 += two_column_buffer[6] * ksrc[44];
        sum_0 += two_column_buffer[5] * ksrc[45];
        sum_1 += two_column_buffer[6] * ksrc[45];
        sum_2 += two_column_buffer[7] * ksrc[45];
        sum_0 += two_column_buffer[6] * ksrc[46];
        sum_1 += two_column_buffer[7] * ksrc[46];
        sum_2 += two_column_buffer[8] * ksrc[46];
        sum_0 += two_column_buffer[7] * ksrc[47];
        sum_1 += two_column_buffer[8] * ksrc[47];
        sum_2 += two_column_buffer[9] * ksrc[47];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[48];
        sum_1 += two_column_buffer[1] * ksrc[48];
        sum_2 += two_column_buffer[2] * ksrc[48];
        sum_0 += two_column_buffer[1] * ksrc[49];
        sum_1 += two_column_buffer[2] * ksrc[49];
        sum_2 += two_column_buffer[3] * ksrc[49];
        sum_0 += two_column_buffer[2] * ksrc[50];
        sum_1 += two_column_buffer[3] * ksrc[50];
        sum_2 += two_column_buffer[4] * ksrc[50];
        sum_0 += two_column_buffer[3] * ksrc[51];
        sum_1 += two_column_buffer[4] * ksrc[51];
        sum_2 += two_column_buffer[5] * ksrc[51];
        sum_0 += two_column_buffer[4] * ksrc[52];
        sum_1 += two_column_buffer[5] * ksrc[52];
        sum_2 += two_column_buffer[6] * ksrc[52];
        sum_0 += two_column_buffer[5] * ksrc[53];
        sum_1 += two_column_buffer[6] * ksrc[53];
        sum_2 += two_column_buffer[7] * ksrc[53];
        sum_0 += two_column_buffer[6] * ksrc[54];
        sum_1 += two_column_buffer[7] * ksrc[54];
        sum_2 += two_column_buffer[8] * ksrc[54];
        sum_0 += two_column_buffer[7] * ksrc[55];
        sum_1 += two_column_buffer[8] * ksrc[55];
        sum_2 += two_column_buffer[9] * ksrc[55];
        two_column_buffer += DIM_KER_Y + 2;

        sum_0 += two_column_buffer[0] * ksrc[56];
        sum_1 += two_column_buffer[1] * ksrc[56];
        sum_2 += two_column_buffer[2] * ksrc[56];
        sum_0 += two_column_buffer[1] * ksrc[57];
        sum_1 += two_column_buffer[2] * ksrc[57];
        sum_2 += two_column_buffer[3] * ksrc[57];
        sum_0 += two_column_buffer[2] * ksrc[58];
        sum_1 += two_column_buffer[3] * ksrc[58];
        sum_2 += two_column_buffer[4] * ksrc[58];
        sum_0 += two_column_buffer[3] * ksrc[59];
        sum_1 += two_column_buffer[4] * ksrc[59];
        sum_2 += two_column_buffer[5] * ksrc[59];
        sum_0 += two_column_buffer[4] * ksrc[60];
        sum_1 += two_column_buffer[5] * ksrc[60];
        sum_2 += two_column_buffer[6] * ksrc[60];
        sum_0 += two_column_buffer[5] * ksrc[61];
        sum_1 += two_column_buffer[6] * ksrc[61];
        sum_2 += two_column_buffer[7] * ksrc[61];
        sum_0 += two_column_buffer[6] * ksrc[62];
        sum_1 += two_column_buffer[7] * ksrc[62];
        sum_2 += two_column_buffer[8] * ksrc[62];
        sum_0 += two_column_buffer[7] * ksrc[63];
        sum_1 += two_column_buffer[8] * ksrc[63];
        sum_2 += two_column_buffer[9] * ksrc[63];

        output_weight_data[(i * output_width + j) * output_depth] -= MIN(MAX(sum_0, output_activation_min), output_activation_max) * scales[0] * learning_rate;
        output_weight_data[(i * output_width + j + 1) * output_depth] -= MIN(MAX(sum_1, output_activation_min), output_activation_max) * scales[0] * learning_rate;
        output_weight_data[(i * output_width + j + 2) * output_depth] -= MIN(MAX(sum_2, output_activation_min), output_activation_max) * scales[0] * learning_rate;
      }

      two_column_buffer_start += DIM_KER_Y + 2;
    }

    ksrc++;
    input_data++;
  }
}
