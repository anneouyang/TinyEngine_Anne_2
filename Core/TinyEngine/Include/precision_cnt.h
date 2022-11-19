/*
 * precision_cnt.h
 *
 *  Created on: Jun 30, 2020
 *      Author: Meenc
 */

#ifndef TINYENGINE_SOURCE_CONVOLUTIONFUNCTIONS_MIX_PRECISION_CNT_H_
#define TINYENGINE_SOURCE_CONVOLUTIONFUNCTIONS_MIX_PRECISION_CNT_H_

/* MIX precision */
#define INPUT_PRE 8
#define KERNEL_PRE 8
#define OUTPUT_PRE 8
#define input_scaler (8 / INPUT_PRE)
#define weight_scaler (8 / KERNEL_PRE)
#define output_scaler (8 / OUTPUT_PRE)


#endif /* TINYENGINE_SOURCE_CONVOLUTIONFUNCTIONS_MIX_PRECISION_CNT_H_ */
