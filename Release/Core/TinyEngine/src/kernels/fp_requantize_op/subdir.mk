################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.c \
../Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.c \
../Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.c \
../Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.c \
../Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.c \
../Core/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.c \
../Core/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.c 

C_DEPS += \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.d \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.d \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.d \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.d \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.d \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.d \
./Core/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.d 

OBJS += \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.o \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.o \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.o \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.o \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.o \
./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.o \
./Core/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.o 


# Each subdirectory must supply rules for building sources it contributes
Core/TinyEngine/src/kernels/fp_requantize_op/%.o Core/TinyEngine/src/kernels/fp_requantize_op/%.su: ../Core/TinyEngine/src/kernels/fp_requantize_op/%.c Core/TinyEngine/src/kernels/fp_requantize_op/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-TinyEngine-2f-src-2f-kernels-2f-fp_requantize_op

clean-Core-2f-TinyEngine-2f-src-2f-kernels-2f-fp_requantize_op:
	-$(RM) ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch16_fpreq.su ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch24_fpreq.su ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch48_fpreq.su ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_ch8_fpreq.su ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_1x1_s8_fpreq.su ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/convolve_s8_kernel3_inputch3_stride2_pad1_fpreq.su ./Core/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.d ./Core/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.o ./Core/TinyEngine/src/kernels/fp_requantize_op/mat_mul_kernels_fpreq.su

.PHONY: clean-Core-2f-TinyEngine-2f-src-2f-kernels-2f-fp_requantize_op

