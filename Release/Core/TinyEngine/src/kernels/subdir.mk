################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/TinyEngine/src/kernels/add.c \
../Core/TinyEngine/src/kernels/arm_convolve_s8.c \
../Core/TinyEngine/src/kernels/arm_depthwise_conv_s8_opt.c \
../Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16.c \
../Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16_reordered.c \
../Core/TinyEngine/src/kernels/arm_q7_to_q15_with_offset.c \
../Core/TinyEngine/src/kernels/avgpooling.c \
../Core/TinyEngine/src/kernels/concat_ch.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_SRAM.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_ch16.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_ch24.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_ch48.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_ch8.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_kbuf.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_oddch.c \
../Core/TinyEngine/src/kernels/convolve_1x1_s8_skip_pad.c \
../Core/TinyEngine/src/kernels/convolve_s8_kernel2x3_inputch3_stride2_pad1.c \
../Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride1_pad1.c \
../Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride2_pad1.c \
../Core/TinyEngine/src/kernels/convolve_s8_kernel3_stride1_pad1.c \
../Core/TinyEngine/src/kernels/convolve_s8_kernel3x2_inputch3_stride2_pad1.c \
../Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride1_pad1.c \
../Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride2_pad1.c \
../Core/TinyEngine/src/kernels/element_mult.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC_inplace.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC_inplace.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC_inplace.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC_inplace.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC_inplace.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC.c \
../Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC_inplace.c \
../Core/TinyEngine/src/kernels/fully_connected.c \
../Core/TinyEngine/src/kernels/mat_mul_fp.c \
../Core/TinyEngine/src/kernels/mat_mult_kernels.c \
../Core/TinyEngine/src/kernels/maxpooling.c \
../Core/TinyEngine/src/kernels/patchpadding_convolve_s8_kernel3_inputch3_stride2.c \
../Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride1_inplace_CHW.c \
../Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride2_inplace_CHW.c \
../Core/TinyEngine/src/kernels/patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2.c \
../Core/TinyEngine/src/kernels/stable_softmax.c \
../Core/TinyEngine/src/kernels/upsample_byte.c 

C_DEPS += \
./Core/TinyEngine/src/kernels/add.d \
./Core/TinyEngine/src/kernels/arm_convolve_s8.d \
./Core/TinyEngine/src/kernels/arm_depthwise_conv_s8_opt.d \
./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16.d \
./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16_reordered.d \
./Core/TinyEngine/src/kernels/arm_q7_to_q15_with_offset.d \
./Core/TinyEngine/src/kernels/avgpooling.d \
./Core/TinyEngine/src/kernels/concat_ch.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_SRAM.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch16.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch24.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch48.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch8.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_kbuf.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_oddch.d \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_skip_pad.d \
./Core/TinyEngine/src/kernels/convolve_s8_kernel2x3_inputch3_stride2_pad1.d \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride1_pad1.d \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride2_pad1.d \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3_stride1_pad1.d \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3x2_inputch3_stride2_pad1.d \
./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride1_pad1.d \
./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride2_pad1.d \
./Core/TinyEngine/src/kernels/element_mult.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC_inplace.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC_inplace.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC_inplace.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC_inplace.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC_inplace.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC.d \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC_inplace.d \
./Core/TinyEngine/src/kernels/fully_connected.d \
./Core/TinyEngine/src/kernels/mat_mul_fp.d \
./Core/TinyEngine/src/kernels/mat_mult_kernels.d \
./Core/TinyEngine/src/kernels/maxpooling.d \
./Core/TinyEngine/src/kernels/patchpadding_convolve_s8_kernel3_inputch3_stride2.d \
./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride1_inplace_CHW.d \
./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride2_inplace_CHW.d \
./Core/TinyEngine/src/kernels/patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2.d \
./Core/TinyEngine/src/kernels/stable_softmax.d \
./Core/TinyEngine/src/kernels/upsample_byte.d 

OBJS += \
./Core/TinyEngine/src/kernels/add.o \
./Core/TinyEngine/src/kernels/arm_convolve_s8.o \
./Core/TinyEngine/src/kernels/arm_depthwise_conv_s8_opt.o \
./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16.o \
./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16_reordered.o \
./Core/TinyEngine/src/kernels/arm_q7_to_q15_with_offset.o \
./Core/TinyEngine/src/kernels/avgpooling.o \
./Core/TinyEngine/src/kernels/concat_ch.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_SRAM.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch16.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch24.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch48.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch8.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_kbuf.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_oddch.o \
./Core/TinyEngine/src/kernels/convolve_1x1_s8_skip_pad.o \
./Core/TinyEngine/src/kernels/convolve_s8_kernel2x3_inputch3_stride2_pad1.o \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride1_pad1.o \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride2_pad1.o \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3_stride1_pad1.o \
./Core/TinyEngine/src/kernels/convolve_s8_kernel3x2_inputch3_stride2_pad1.o \
./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride1_pad1.o \
./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride2_pad1.o \
./Core/TinyEngine/src/kernels/element_mult.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC_inplace.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC_inplace.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC_inplace.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC_inplace.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC_inplace.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC.o \
./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC_inplace.o \
./Core/TinyEngine/src/kernels/fully_connected.o \
./Core/TinyEngine/src/kernels/mat_mul_fp.o \
./Core/TinyEngine/src/kernels/mat_mult_kernels.o \
./Core/TinyEngine/src/kernels/maxpooling.o \
./Core/TinyEngine/src/kernels/patchpadding_convolve_s8_kernel3_inputch3_stride2.o \
./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride1_inplace_CHW.o \
./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride2_inplace_CHW.o \
./Core/TinyEngine/src/kernels/patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2.o \
./Core/TinyEngine/src/kernels/stable_softmax.o \
./Core/TinyEngine/src/kernels/upsample_byte.o 


# Each subdirectory must supply rules for building sources it contributes
Core/TinyEngine/src/kernels/%.o Core/TinyEngine/src/kernels/%.su: ../Core/TinyEngine/src/kernels/%.c Core/TinyEngine/src/kernels/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-TinyEngine-2f-src-2f-kernels

clean-Core-2f-TinyEngine-2f-src-2f-kernels:
	-$(RM) ./Core/TinyEngine/src/kernels/add.d ./Core/TinyEngine/src/kernels/add.o ./Core/TinyEngine/src/kernels/add.su ./Core/TinyEngine/src/kernels/arm_convolve_s8.d ./Core/TinyEngine/src/kernels/arm_convolve_s8.o ./Core/TinyEngine/src/kernels/arm_convolve_s8.su ./Core/TinyEngine/src/kernels/arm_depthwise_conv_s8_opt.d ./Core/TinyEngine/src/kernels/arm_depthwise_conv_s8_opt.o ./Core/TinyEngine/src/kernels/arm_depthwise_conv_s8_opt.su ./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16.d ./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16.o ./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16.su ./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16_reordered.d ./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16_reordered.o ./Core/TinyEngine/src/kernels/arm_nn_mat_mult_kernel_s8_s16_reordered.su ./Core/TinyEngine/src/kernels/arm_q7_to_q15_with_offset.d ./Core/TinyEngine/src/kernels/arm_q7_to_q15_with_offset.o ./Core/TinyEngine/src/kernels/arm_q7_to_q15_with_offset.su ./Core/TinyEngine/src/kernels/avgpooling.d ./Core/TinyEngine/src/kernels/avgpooling.o ./Core/TinyEngine/src/kernels/avgpooling.su ./Core/TinyEngine/src/kernels/concat_ch.d ./Core/TinyEngine/src/kernels/concat_ch.o ./Core/TinyEngine/src/kernels/concat_ch.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_SRAM.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_SRAM.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_SRAM.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch16.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch16.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch16.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch24.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch24.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch24.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch48.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch48.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch48.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch8.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch8.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_ch8.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_kbuf.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_kbuf.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_kbuf.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_oddch.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_oddch.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_oddch.su ./Core/TinyEngine/src/kernels/convolve_1x1_s8_skip_pad.d ./Core/TinyEngine/src/kernels/convolve_1x1_s8_skip_pad.o ./Core/TinyEngine/src/kernels/convolve_1x1_s8_skip_pad.su ./Core/TinyEngine/src/kernels/convolve_s8_kernel2x3_inputch3_stride2_pad1.d ./Core/TinyEngine/src/kernels/convolve_s8_kernel2x3_inputch3_stride2_pad1.o ./Core/TinyEngine/src/kernels/convolve_s8_kernel2x3_inputch3_stride2_pad1.su ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride1_pad1.d ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride1_pad1.o ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride1_pad1.su ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride2_pad1.d ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride2_pad1.o ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_inputch3_stride2_pad1.su ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_stride1_pad1.d ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_stride1_pad1.o ./Core/TinyEngine/src/kernels/convolve_s8_kernel3_stride1_pad1.su ./Core/TinyEngine/src/kernels/convolve_s8_kernel3x2_inputch3_stride2_pad1.d ./Core/TinyEngine/src/kernels/convolve_s8_kernel3x2_inputch3_stride2_pad1.o ./Core/TinyEngine/src/kernels/convolve_s8_kernel3x2_inputch3_stride2_pad1.su ./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride1_pad1.d ./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride1_pad1.o ./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride1_pad1.su ./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride2_pad1.d ./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride2_pad1.o ./Core/TinyEngine/src/kernels/convolve_u8_kernel3_inputch3_stride2_pad1.su ./Core/TinyEngine/src/kernels/element_mult.d ./Core/TinyEngine/src/kernels/element_mult.o ./Core/TinyEngine/src/kernels/element_mult.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC_inplace.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC_inplace.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride1_pad1_a8w8_8bit_HWC_inplace.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC_inplace.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC_inplace.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel3_stride2_pad1_a8w8_8bit_HWC_inplace.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC_inplace.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC_inplace.o
	-$(RM) ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride1_pad2_a8w8_8bit_HWC_inplace.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC_inplace.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC_inplace.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel5_stride2_pad2_a8w8_8bit_HWC_inplace.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC_inplace.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC_inplace.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride1_pad3_a8w8_8bit_HWC_inplace.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC.su ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC_inplace.d ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC_inplace.o ./Core/TinyEngine/src/kernels/fast_depthwise_conv_s8_kernel7_stride2_pad3_a8w8_8bit_HWC_inplace.su ./Core/TinyEngine/src/kernels/fully_connected.d ./Core/TinyEngine/src/kernels/fully_connected.o ./Core/TinyEngine/src/kernels/fully_connected.su ./Core/TinyEngine/src/kernels/mat_mul_fp.d ./Core/TinyEngine/src/kernels/mat_mul_fp.o ./Core/TinyEngine/src/kernels/mat_mul_fp.su ./Core/TinyEngine/src/kernels/mat_mult_kernels.d ./Core/TinyEngine/src/kernels/mat_mult_kernels.o ./Core/TinyEngine/src/kernels/mat_mult_kernels.su ./Core/TinyEngine/src/kernels/maxpooling.d ./Core/TinyEngine/src/kernels/maxpooling.o ./Core/TinyEngine/src/kernels/maxpooling.su ./Core/TinyEngine/src/kernels/patchpadding_convolve_s8_kernel3_inputch3_stride2.d ./Core/TinyEngine/src/kernels/patchpadding_convolve_s8_kernel3_inputch3_stride2.o ./Core/TinyEngine/src/kernels/patchpadding_convolve_s8_kernel3_inputch3_stride2.su ./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride1_inplace_CHW.d ./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride1_inplace_CHW.o ./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride1_inplace_CHW.su ./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride2_inplace_CHW.d ./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride2_inplace_CHW.o ./Core/TinyEngine/src/kernels/patchpadding_depthwise_kernel3x3_stride2_inplace_CHW.su ./Core/TinyEngine/src/kernels/patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2.d ./Core/TinyEngine/src/kernels/patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2.o ./Core/TinyEngine/src/kernels/patchpadding_kbuf_convolve_s8_kernel3_inputch3_stride2.su ./Core/TinyEngine/src/kernels/stable_softmax.d ./Core/TinyEngine/src/kernels/stable_softmax.o ./Core/TinyEngine/src/kernels/stable_softmax.su ./Core/TinyEngine/src/kernels/upsample_byte.d ./Core/TinyEngine/src/kernels/upsample_byte.o ./Core/TinyEngine/src/kernels/upsample_byte.su

.PHONY: clean-Core-2f-TinyEngine-2f-src-2f-kernels

