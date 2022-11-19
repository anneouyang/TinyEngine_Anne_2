/* Automatically generated source file */
#include "arm_nnfunctions.h"

#include "genNN.h"
#include "genModel.h"

#include "tinyengine_function.h"


/* Variables used by all ops */
ADD_params add_params;

signed char* getInput() {
	return buffer0;
}
signed char* getOutput() {
	return NNoutput;
}
void end2endinference(q7_t* img){
    invoke();
}
void invoke(){
/* layer 0:CONV_2D */
convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[0],80,80,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,1,-128,127,&buffer0[64000],40,40,16,sbuf,kbuf,-1);
/* layer 1:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[64000],40,40,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-128,128,-128,127,&buffer0[0],40,40,16,sbuf,-128);
/* layer 2:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[64000],40,40,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-2,128,-128,127,&buffer0[0],40,40,8,sbuf);
/* layer 3:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[0],40,40,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-128,2,-128,127,&buffer0[12800],40,40,48,sbuf);
/* layer 4:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[12800],40,40,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-128,128,-128,127,&buffer0[0],20,20,48,sbuf,-128);
/* layer 5:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[12800],20,20,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-19,128,-128,127,&buffer1[0],20,20,16,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_ch16(&buffer1[0],20,20,16,(const q7_t*) weight6,bias6,shift6,multiplier6,-128,19,-128,127,&buffer0[70400],20,20,48,sbuf);
/* layer 7:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[70400],20,20,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-128,128,-128,127,&buffer0[0],20,20,48,sbuf,-128);
/* layer 8:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[70400],20,20,48,(const q7_t*) weight8,bias8,shift8,multiplier8,12,128,-128,127,&buffer0[0],20,20,16,sbuf);
/* layer 9:ADD */
add_params.input_h = 20;add_params.input_w = 20;add_params.input_c = 16;
add_params.left_shift = 20;add_params.input1_offset = -12;add_params.input1_multiplier = 1853017153;add_params.input1_shift = -1;add_params.input2_offset = 19;add_params.input2_multiplier = 1073741824;add_params.input2_shift = 0;add_params.output_offset = -22;add_params.output_multiplier = 1738872347;add_params.output_shift = -19;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer1[0]);
/* layer 10:CONV_2D */
convolve_1x1_s8_ch16(&buffer1[0],20,20,16,(const q7_t*) weight9,bias9,shift9,multiplier9,-128,22,-128,127,&buffer0[70400],20,20,48,sbuf);
/* layer 11:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[70400],20,20,48,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-128,128,-128,127,&buffer0[0],20,20,48,sbuf,-128);
/* layer 12:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[70400],20,20,48,(const q7_t*) weight11,bias11,shift11,multiplier11,-10,128,-128,127,&buffer0[0],20,20,16,sbuf);
/* layer 13:ADD */
add_params.input_h = 20;add_params.input_w = 20;add_params.input_c = 16;
add_params.left_shift = 20;add_params.input1_offset = 10;add_params.input1_multiplier = 1983546124;add_params.input1_shift = -3;add_params.input2_offset = 22;add_params.input2_multiplier = 1073741824;add_params.input2_shift = 0;add_params.output_offset = -14;add_params.output_multiplier = 2104562937;add_params.output_shift = -19;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer0[83200]);
/* layer 14:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[83200],20,20,16,(const q7_t*) weight12,bias12,shift12,multiplier12,-128,14,-128,127,&buffer0[0],20,20,48,sbuf);
/* layer 15:DEPTHWISE_CONV_2D */
depthwise_kernel7x7_stride2_inplace_CHW(&buffer0[0],20,20,48,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-128,128,-128,127,&buffer0[84800],10,10,48,sbuf,-128);
/* layer 16:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],10,10,48,(const q7_t*) weight14,bias14,shift14,multiplier14,2,128,-128,127,&buffer1[0],10,10,24,sbuf);
/* layer 17:CONV_2D */
convolve_1x1_s8_ch24(&buffer1[0],10,10,24,(const q7_t*) weight15,bias15,shift15,multiplier15,-128,-2,-128,127,&buffer0[75200],10,10,144,sbuf);
/* layer 18:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[75200],10,10,144,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-128,128,-128,127,&buffer0[0],10,10,144,sbuf,-128);
/* layer 19:CONV_2D */
convolve_1x1_s8(&buffer0[75200],10,10,144,(const q7_t*) weight17,bias17,shift17,multiplier17,2,128,-128,127,&buffer0[0],10,10,24,sbuf);
/* layer 20:ADD */
add_params.input_h = 10;add_params.input_w = 10;add_params.input_c = 24;
add_params.left_shift = 20;add_params.input1_offset = -2;add_params.input1_multiplier = 1073741824;add_params.input1_shift = 0;add_params.input2_offset = -2;add_params.input2_multiplier = 2067874274;add_params.input2_shift = -1;add_params.output_offset = 12;add_params.output_multiplier = 1844006744;add_params.output_shift = -19;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer1[0]);
/* layer 21:CONV_2D */
convolve_1x1_s8_ch24(&buffer1[0],10,10,24,(const q7_t*) weight18,bias18,shift18,multiplier18,-128,-12,-128,127,&buffer0[77600],10,10,120,sbuf);
/* layer 22:DEPTHWISE_CONV_2D */
depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[77600],10,10,120,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-128,128,-128,127,&buffer0[0],10,10,120,sbuf,-128);
/* layer 23:CONV_2D */
convolve_1x1_s8(&buffer0[77600],10,10,120,(const q7_t*) weight20,bias20,shift20,multiplier20,0,128,-128,127,&buffer0[0],10,10,24,sbuf);
/* layer 24:ADD */
add_params.input_h = 10;add_params.input_w = 10;add_params.input_c = 24;
add_params.left_shift = 20;add_params.input1_offset = 0;add_params.input1_multiplier = 2142432871;add_params.input1_shift = -3;add_params.input2_offset = -12;add_params.input2_multiplier = 1073741824;add_params.input2_shift = 0;add_params.output_offset = 18;add_params.output_multiplier = 1986712119;add_params.output_shift = -19;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer0[87200]);
/* layer 25:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[87200],10,10,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-128,-18,-128,127,&buffer0[0],10,10,144,sbuf);
/* layer 26:DEPTHWISE_CONV_2D */
depthwise_kernel7x7_stride2_inplace_CHW(&buffer0[0],10,10,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-128,128,-128,127,&buffer0[86000],5,5,144,sbuf,-128);
/* layer 27:CONV_2D */
convolve_1x1_s8(&buffer0[0],5,5,144,(const q7_t*) weight23,bias23,shift23,multiplier23,-11,128,-128,127,&buffer1[0],5,5,40,sbuf);
/* layer 28:CONV_2D */
convolve_1x1_s8(&buffer1[0],5,5,40,(const q7_t*) weight24,bias24,shift24,multiplier24,-128,11,-128,127,&buffer0[83600],5,5,240,sbuf);
/* layer 29:DEPTHWISE_CONV_2D */
depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[83600],5,5,240,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,-128,128,-128,127,&buffer0[0],5,5,240,sbuf,-128);
/* layer 30:CONV_2D */
convolve_1x1_s8(&buffer0[83600],5,5,240,(const q7_t*) weight26,bias26,shift26,multiplier26,7,128,-128,127,&buffer0[0],5,5,40,sbuf);
/* layer 31:ADD */
add_params.input_h = 5;add_params.input_w = 5;add_params.input_c = 40;
add_params.left_shift = 20;add_params.input1_offset = -7;add_params.input1_multiplier = 1626712137;add_params.input1_shift = -1;add_params.input2_offset = 11;add_params.input2_multiplier = 1073741824;add_params.input2_shift = 0;add_params.output_offset = -3;add_params.output_multiplier = 1074737776;add_params.output_shift = -18;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer0[88600]);
/* layer 32:CONV_2D */
convolve_1x1_s8(&buffer0[88600],5,5,40,(const q7_t*) weight27,bias27,shift27,multiplier27,-128,3,-128,127,&buffer0[0],5,5,240,sbuf);
/* layer 33:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,240,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-128,128,-128,127,&buffer0[83600],5,5,240,sbuf,-128);
/* layer 34:CONV_2D */
convolve_1x1_s8(&buffer0[0],5,5,240,(const q7_t*) weight29,bias29,shift29,multiplier29,-5,128,-128,127,&buffer1[0],5,5,48,sbuf);
/* layer 35:CONV_2D */
convolve_1x1_s8_ch48(&buffer1[0],5,5,48,(const q7_t*) weight30,bias30,shift30,multiplier30,-128,5,-128,127,&buffer0[84800],5,5,192,sbuf);
/* layer 36:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[84800],5,5,192,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-128,128,-128,127,&buffer0[0],5,5,192,sbuf,-128);
/* layer 37:CONV_2D */
convolve_1x1_s8(&buffer0[84800],5,5,192,(const q7_t*) weight32,bias32,shift32,multiplier32,-10,128,-128,127,&buffer0[0],5,5,48,sbuf);
/* layer 38:ADD */
add_params.input_h = 5;add_params.input_w = 5;add_params.input_c = 48;
add_params.left_shift = 20;add_params.input1_offset = 10;add_params.input1_multiplier = 1073741824;add_params.input1_shift = 0;add_params.input2_offset = 5;add_params.input2_multiplier = 1728811214;add_params.input2_shift = -1;add_params.output_offset = -2;add_params.output_multiplier = 1092740395;add_params.output_shift = -18;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer0[88400]);
/* layer 39:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[88400],5,5,48,(const q7_t*) weight33,bias33,shift33,multiplier33,-128,2,-128,127,&buffer0[0],5,5,240,sbuf);
/* layer 40:DEPTHWISE_CONV_2D */
depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],5,5,240,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-128,128,-128,127,&buffer0[87440],3,3,240,sbuf,-128);
/* layer 41:CONV_2D */
convolve_1x1_s8(&buffer0[0],3,3,240,(const q7_t*) weight35,bias35,shift35,multiplier35,-8,128,-128,127,&buffer1[0],3,3,96,sbuf);
/* layer 42:CONV_2D */
convolve_1x1_s8(&buffer1[0],3,3,96,(const q7_t*) weight36,bias36,shift36,multiplier36,-128,8,-128,127,&buffer0[85280],3,3,480,sbuf);
/* layer 43:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[85280],3,3,480,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,-128,128,-128,127,&buffer0[0],3,3,480,sbuf,-128);
/* layer 44:CONV_2D */
convolve_1x1_s8(&buffer0[85280],3,3,480,(const q7_t*) weight38,bias38,shift38,multiplier38,14,128,-128,127,&buffer0[0],3,3,96,sbuf);
/* layer 45:ADD */
add_params.input_h = 3;add_params.input_w = 3;add_params.input_c = 96;
add_params.left_shift = 20;add_params.input1_offset = -14;add_params.input1_multiplier = 1931955708;add_params.input1_shift = -1;add_params.input2_offset = 8;add_params.input2_multiplier = 1073741824;add_params.input2_shift = 0;add_params.output_offset = 9;add_params.output_multiplier = 2125452290;add_params.output_shift = -19;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer1[0]);
/* layer 46:CONV_2D */
convolve_1x1_s8(&buffer1[0],3,3,96,(const q7_t*) weight39,bias39,shift39,multiplier39,-128,-9,-128,127,&buffer0[86144],3,3,384,sbuf);
/* layer 47:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[86144],3,3,384,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-128,128,-128,127,&buffer0[0],3,3,384,sbuf,-128);
/* layer 48:CONV_2D */
convolve_1x1_s8(&buffer0[86144],3,3,384,(const q7_t*) weight41,bias41,shift41,multiplier41,1,128,-128,127,&buffer0[0],3,3,96,sbuf);
/* layer 49:ADD */
add_params.input_h = 3;add_params.input_w = 3;add_params.input_c = 96;
add_params.left_shift = 20;add_params.input1_offset = -1;add_params.input1_multiplier = 1890466689;add_params.input1_shift = -2;add_params.input2_offset = -9;add_params.input2_multiplier = 1073741824;add_params.input2_shift = 0;add_params.output_offset = 7;add_params.output_multiplier = 1887616697;add_params.output_shift = -19;add_params.quantized_activation_max = 127;add_params.quantized_activation_min = -128;
add(add_params.input_c * add_params.input_h * add_params.input_w, &add_params, &buffer0[0], &buffer1[0], &buffer0[88736]);
/* layer 50:CONV_2D */
convolve_1x1_s8(&buffer0[88736],3,3,96,(const q7_t*) weight42,bias42,shift42,multiplier42,-128,-7,-128,127,&buffer0[0],3,3,288,sbuf);
/* layer 51:DEPTHWISE_CONV_2D */
depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],3,3,288,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-128,128,-128,127,&buffer0[87008],3,3,288,sbuf,-128);
/* layer 52:CONV_2D */
convolve_1x1_s8(&buffer0[0],3,3,288,(const q7_t*) weight44,bias44,shift44,multiplier44,6,128,-128,127,&buffer0[88160],3,3,160,sbuf);
/* layer 53:AVERAGE_POOL_2D */
avg_pooling(&buffer0[88160],3,3,160,3,3,1,1,-128,127,&buffer0[0]);
/* layer 54:CONV_2D */
convolve_1x1_s8(&buffer0[0],1,1,160,(const q7_t*) weight45,bias45,shift45,multiplier45,0,-6,-128,127,&buffer0[89598],1,1,2,sbuf);
}
