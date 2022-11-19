################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW_fpreq.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW_fpreq.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW.c \
../Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.c \
../Core/TinyEngine/codegen/Source/genModel.c 

C_DEPS += \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW_fpreq.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW_fpreq.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW.d \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.d \
./Core/TinyEngine/codegen/Source/genModel.d 

OBJS += \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW_fpreq.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW_fpreq.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW.o \
./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.o \
./Core/TinyEngine/codegen/Source/genModel.o 


# Each subdirectory must supply rules for building sources it contributes
Core/TinyEngine/codegen/Source/%.o Core/TinyEngine/codegen/Source/%.su: ../Core/TinyEngine/codegen/Source/%.c Core/TinyEngine/codegen/Source/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-TinyEngine-2f-codegen-2f-Source

clean-Core-2f-TinyEngine-2f-codegen-2f-Source:
	-$(RM) ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW.d ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW.o ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW.su ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.d ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.o ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride1_inplace_CHW_fpreq.su ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW.d ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW.o ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW.su ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW_fpreq.d ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW_fpreq.o ./Core/TinyEngine/codegen/Source/depthwise_kernel3x3_stride2_inplace_CHW_fpreq.su ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW.d ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW.o ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW.su ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.d ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.o ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride1_inplace_CHW_fpreq.su ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW.d ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW.o ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW.su ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW_fpreq.d ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW_fpreq.o ./Core/TinyEngine/codegen/Source/depthwise_kernel5x5_stride2_inplace_CHW_fpreq.su ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW.d ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW.o ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW.su ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.d ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.o ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride1_inplace_CHW_fpreq.su ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW.d ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW.o ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW.su ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.d ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.o ./Core/TinyEngine/codegen/Source/depthwise_kernel7x7_stride2_inplace_CHW_fpreq.su ./Core/TinyEngine/codegen/Source/genModel.d ./Core/TinyEngine/codegen/Source/genModel.o ./Core/TinyEngine/codegen/Source/genModel.su

.PHONY: clean-Core-2f-TinyEngine-2f-codegen-2f-Source

