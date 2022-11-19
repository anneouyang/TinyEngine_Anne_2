/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author: 
 * - Wei-Ming Chen, wmchen@mit.edu
 * - Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

// TODO: Could have errors when input_channel % 4 != 0

/************* START: 3*3 Kernel Functions *************/
inline void load_3row_3col_fp(const float* src, const float* src2, const float* src3, float* dst, float* dst2, float* dst3, const int channel_div4) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;

    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
  }
}

inline void load_3row_2col_fp(const float* src, const float* src2, const float* src3, float* dst, float* dst2, float* dst3, const int channel_div4) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;

    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
    *dst3++ = *src3++;
  }
}

inline void load_2row_3col_fp(const float* src, const float* src2, float* dst, float* dst2, const int channel_div4) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
  }
}

inline void load_2row_2col_fp(const float* src, const float* src2, float* dst, float* dst2, const int channel_div4) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
    *dst2++ = *src2++;
  }
}

inline void pad_3row_1col_fp(float* dst, float* dst2, float* dst3, const int channel_div4, const int pad_value) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = pad_value;
    *dst++ = pad_value;
    *dst++ = pad_value;
    *dst++ = pad_value;

    *dst2++ = pad_value;
    *dst2++ = pad_value;
    *dst2++ = pad_value;
    *dst2++ = pad_value;

    *dst3++ = pad_value;
    *dst3++ = pad_value;
    *dst3++ = pad_value;
    *dst3++ = pad_value;
  }
}

inline void pad_2row_1col_fp(float* dst, float* dst2, const int channel_div4, const int pad_value) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = pad_value;
    *dst++ = pad_value;
    *dst++ = pad_value;
    *dst++ = pad_value;

    *dst2++ = pad_value;
    *dst2++ = pad_value;
    *dst2++ = pad_value;
    *dst2++ = pad_value;
  }
}

inline void pad_1row_3col_fp(float* dst, float* dst2, float* dst3, const int channel_div4, const int pad_value) {
  int cnt = channel_div4;

  while (cnt--) {
    *dst++ = pad_value;
    *dst++ = pad_value;
    *dst++ = pad_value;
    *dst++ = pad_value;

    *dst2++ = pad_value;
    *dst2++ = pad_value;
    *dst2++ = pad_value;
    *dst2++ = pad_value;

    *dst3++ = pad_value;
    *dst3++ = pad_value;
    *dst3++ = pad_value;
    *dst3++ = pad_value;
  }
}
/************* END: 3*3 Kernel Functions *************/


/************* START: x*x Kernel Functions *************/
inline void load_xrow_xcol_fp(const float* src, float* dst, const int input_width, const int input_depth, 
                              const int load_col_width, const int filter_width, const int filter_height) {
  const float* src_start = src;
  float* dst_start = dst;
  const int in_row_offset = input_depth * input_width;
  const int channel_div4 = input_depth >> 2;
  const int load_col_width_div4 = load_col_width >> 2;
  int i;

  for (i = 0; i < filter_height; i++) {
    src = src_start + i * in_row_offset;
    dst = dst_start + i * filter_width * input_depth;

    int block_cnt = channel_div4 * load_col_width_div4;
    while (block_cnt--) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //4
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //8
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //12
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //16
    }

    int leftover_block_cnt = channel_div4 * (load_col_width & 0x3);
    while (leftover_block_cnt--) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
    }
  }
}

inline void group_load_xrow_xcol_fp(const float* src, float* dst, const int input_width, const int input_depth, const int filter_depth, 
                              const int load_col_width, const int filter_width, const int filter_height) {
  const float* src_start = src;
  float* dst_start = dst;
  const int in_row_offset = input_depth * input_width;
  const int channel_div4 = filter_depth >> 2;
  const int load_col_width_div4 = load_col_width >> 2;
  int i;

  for (i = 0; i < filter_height; i++) {
    src = src_start + i * in_row_offset;
    dst = dst_start + i * filter_width * filter_depth;

    int block_cnt = channel_div4 * load_col_width_div4;
    while (block_cnt--) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //4
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //8
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //12
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++; //16
    }

    int leftover_block_cnt = channel_div4 * (load_col_width & 0x3);
    while (leftover_block_cnt--) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
    }
  }
}

inline void pad_xrow_1col_fp(float* dst, const int input_depth, const int filter_width, const int filter_height, const int pad_value) {
  float* dst_start = dst;
  const int channel_div4 = input_depth >> 2;
  int i;

  for (i = 0; i < filter_height; i++) {
    dst = dst_start + i * filter_width * input_depth;

    int block_cnt = channel_div4;
    while (block_cnt--) {
      *dst++ = pad_value;
      *dst++ = pad_value;
      *dst++ = pad_value;
      *dst++ = pad_value;
    }
  }
}
/************* END: x*x Kernel Functions *************/
