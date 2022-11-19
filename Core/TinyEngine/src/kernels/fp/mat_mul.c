/* ----------------------------------------------------------------------
 * Name: mat_mul.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define COL_PER_LOOP (8)

tinyengine_status_fp mat_mul(const float* matA, const uint16_t matA_row, const uint16_t matA_col,
				const float* matB, const uint16_t matB_col, float* output)
{
	int m, n, i;
	for (n = 0; n < matA_row; n++){
		for (m = 0; m < matB_col; m++){
			float sum = 0;

			/* Naive Version */
			/*
			for (i = 0; i < matA_col; i++){
				sum += matA[i + n * matA_col] * matB[m + i * matB_col];
			}
			*/

			/* Loop Unrolling Version */
			
			int col_count = matA_col / COL_PER_LOOP;
			if (col_count){
				for (i = 0; col_count>0; i+=COL_PER_LOOP, --col_count){
					sum += matA[i + n * matA_col] * matB[m + i * matB_col];
					sum += matA[(i+1) + n * matA_col] * matB[m + (i+1) * matB_col];
					sum += matA[(i+2) + n * matA_col] * matB[m + (i+2) * matB_col];
					sum += matA[(i+3) + n * matA_col] * matB[m + (i+3) * matB_col];
					sum += matA[(i+4) + n * matA_col] * matB[m + (i+4) * matB_col];
					sum += matA[(i+5) + n * matA_col] * matB[m + (i+5) * matB_col];
					sum += matA[(i+6) + n * matA_col] * matB[m + (i+6) * matB_col];
					sum += matA[(i+7) + n * matA_col] * matB[m + (i+7) * matB_col];
				}
			}
			col_count = matA_col & (COL_PER_LOOP - 1);
			if (col_count){
				for(i = matA_col-1; i >= matA_col-col_count; --i){
					sum += matA[i + n * matA_col] * matB[m + i * matB_col];
				}
			}
			

			output[m + n * matB_col] = sum;
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}
