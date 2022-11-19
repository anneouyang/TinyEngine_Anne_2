/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.ed
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#ifndef INC_GENNN_H_
#define INC_GENNN_H_

#include <stdint.h>

signed char* getInput();
signed char* getOutput();
float* getOutput_fp();
void setupBuffer();
void invoke(float* labels);
void getResult(uint8_t *P, uint8_t *NP);
int* getKbuffer();
void end2endinference();

#endif /* INC_GENNN_H_ */
