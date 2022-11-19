#include "tinyengine_function.h"

tinyengine_status mat_mul_fp(
				const float *matA, const uint16_t matA_row, const uint16_t matA_col,
				const float* matB, const uint16_t matB_col, float* output)
{
	int m, n, i;
	for (n = 0; n < matA_row; n++){
		for (m = 0; m < matB_col; m++){
			float sum = 0;
			for (i = 0; i < matA_col; i++){
				sum += matA[i + n * matA_col] * matB[m + i * matA_col];
			}
			output[m + n * matB_col] = sum;
		}
	}
}
