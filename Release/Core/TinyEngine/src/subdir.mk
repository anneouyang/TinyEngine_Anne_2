################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/TinyEngine/src/detectionUtility.c \
../Core/TinyEngine/src/yoloOutput.c 

C_DEPS += \
./Core/TinyEngine/src/detectionUtility.d \
./Core/TinyEngine/src/yoloOutput.d 

OBJS += \
./Core/TinyEngine/src/detectionUtility.o \
./Core/TinyEngine/src/yoloOutput.o 


# Each subdirectory must supply rules for building sources it contributes
Core/TinyEngine/src/%.o Core/TinyEngine/src/%.su: ../Core/TinyEngine/src/%.c Core/TinyEngine/src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F746xx -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-TinyEngine-2f-src

clean-Core-2f-TinyEngine-2f-src:
	-$(RM) ./Core/TinyEngine/src/detectionUtility.d ./Core/TinyEngine/src/detectionUtility.o ./Core/TinyEngine/src/detectionUtility.su ./Core/TinyEngine/src/yoloOutput.d ./Core/TinyEngine/src/yoloOutput.o ./Core/TinyEngine/src/yoloOutput.su

.PHONY: clean-Core-2f-TinyEngine-2f-src

