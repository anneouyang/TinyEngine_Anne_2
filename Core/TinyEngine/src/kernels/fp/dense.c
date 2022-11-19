/* ----------------------------------------------------------------------
 * Name: dense.c
 * Project: TinyEngine, MCUNetV3
 * Contact author: Wei-Chen Wang, wweichen@mit.edu
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"
#define COL_PER_LOOP (8)

tinyengine_status_fp dense(const float* matA, const uint16_t matA_row, const uint16_t matA_col,
				const float* matB, const uint16_t matB_row, float* output)
{
	int m, n, i;
	for (n = 0; n < matA_row; n++){
		for (m = 0; m < matB_row; m++){
			float sum = 0;

			/* Naive Version */
			/*
			for (i = 0; i < matA_col; i++){
				sum += matA[i + n * matA_col] * matB[i + m * matA_col];
			}
			*/

			/* Loop Unrolling Version */
			int col_count = matA_col / COL_PER_LOOP;
			if (col_count){
				for (i = 0; col_count>0; i+=COL_PER_LOOP, --col_count){
					sum += matA[i + n * matA_col] * matB[i + m * matA_col];
					sum += matA[(i+1) + n * matA_col] * matB[(i+1) + m * matA_col];
					sum += matA[(i+2) + n * matA_col] * matB[(i+2) + m * matA_col];
					sum += matA[(i+3) + n * matA_col] * matB[(i+3) + m * matA_col];
					sum += matA[(i+4) + n * matA_col] * matB[(i+4) + m * matA_col];
					sum += matA[(i+5) + n * matA_col] * matB[(i+5) + m * matA_col];
					sum += matA[(i+6) + n * matA_col] * matB[(i+6) + m * matA_col];
					sum += matA[(i+7) + n * matA_col] * matB[(i+7) + m * matA_col];
				}
			}
			col_count = matA_col & (COL_PER_LOOP - 1);
			if (col_count){
				for(i = matA_col-1; i >= matA_col-col_count; --i){
					sum += matA[i + n * matA_col] * matB[i + m * matA_col];
				}
			}
			

			output[m + n * matB_row] = sum;
		}
	}
	
	/* Return to application */
	return STATE_SUCCESS_fp;
}
