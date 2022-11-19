/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.edu
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#include "main.h"
#include "stdio.h"
#include "../testing_data/images.h"
#include "golden_data.h"
#include "profile.h"
extern "C"{
#include "tinyengine_function.h"
#include "genNN.h"
}

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart1;
UART_HandleTypeDef huart6;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART6_UART_Init(void);


#define IMAGE_H 80
#define IMAGE_W 80
uint16_t color;
void example_VWW(const int8_t* image) {
	signed char *input = getInput();
    int i;
    for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
            input[i] = *image++;
    }
    invoke(NULL);
    uint8_t* output = (uint8_t*)getOutput();
    uint8_t P = output[0], NP = output[1];

    if (P > NP){
    	printf("It's a person");
    }
    else{
    	printf("It's not a person");
    }
}

#define INPUT_CH 160
#define OUTPUT_CH 2
#define IMAGES 6

void StartDefaultTask(void const * argument);
float feat_fp[INPUT_CH];
int8_t feat[INPUT_CH];
float w[INPUT_CH * OUTPUT_CH];
float b[OUTPUT_CH];
float out[OUTPUT_CH];
float dw[OUTPUT_CH*INPUT_CH];
static float lr = 0.1;

const int label[6] = {0, 0, 0, 1, 1, 1};


void invoke_new_weights(const int8_t* img, float *out){
  int i;
  signed char *input = getInput();
  const int8_t* image = img;
  for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
	  input[i] = *image++;
  }
  invoke(NULL);
  signed char *output = getOutput();
  for (i = 0; i < INPUT_CH; i++){
	  feat_fp[i] = (output[i] - zero_x)*scale_x;
  }

  // out = new_w @ feat + new_b
  fully_connected_fp(feat_fp, 1, 1, INPUT_CH, OUTPUT_CH, b, w, out);
}

void train_one_img(const int8_t* img, int cls)
{
  int i;
  signed char *input = getInput();
  const int8_t* image = img;
  for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
	  input[i] = *image++;
  }
  invoke(NULL);
  signed char *output = getOutput();
  for (i = 0; i < INPUT_CH; i++){
	  feat_fp[i] = (output[i] - zero_x)*scale_x;
  }

  // out = new_w @ feat + new_b
  fully_connected_fp(feat_fp, 1, 1, INPUT_CH, OUTPUT_CH, b, w, out);

  // softmax = _stable_softmax(out)
  statble_softmax_inplace(out, OUTPUT_CH);

  out[cls] -= 1;

  //dw = dy.reshape(-1, 1) @ feat.reshape(1, -1)
  mat_mul_fp(out, OUTPUT_CH, 1, feat_fp, INPUT_CH, dw);

  for (i = 0; i < OUTPUT_CH * INPUT_CH; i++){
	  w[i] = w[i] - lr * dw[i];
  }
  //new_w = new_w - lr * dw
  //new_b = new_b - lr *
  b[0] = b[0] - lr * out[0];
  b[1] = b[1] - lr * out[1];
}

void train_one_feat(const float* feat, int cls)
{
  int i;
  signed char *input = getInput();
  for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
	  input[i] = feat[i];
  }

  // out = new_w @ feat + new_b
  fully_connected_fp(feat, 1, 1, INPUT_CH, OUTPUT_CH, b, w, out);

  // softmax = _stable_softmax(out)
  statble_softmax_inplace(out, OUTPUT_CH);

  out[cls] -= 1;

  //dw = dy.reshape(-1, 1) @ feat.reshape(1, -1)
  mat_mul_fp(out, OUTPUT_CH, 1, feat, INPUT_CH, dw);

  for (i = 0; i < OUTPUT_CH * INPUT_CH; i++){
	  w[i] = w[i] - lr * dw[i];
  }
  //new_w = new_w - lr * dw
  //new_b = new_b - lr *
  b[0] = b[0] - lr * out[0];
  b[1] = b[1] - lr * out[1];
}

//#define TESTFP 1
#define TESTINT8 1

int main(void)
{
  sprintf(buf,"********* START *********\r\n");
  printLog(buf);

  int i;
  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  HAL_Init();

  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_USART6_UART_Init();

#ifdef TESTFP
    printLog("** FP32 Start **\r\n");
	float *input = (float *)getInput();
    for (i = 0; i < 1*32*32;i++){
  	  input[i] = 1;
    }
    float labels[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    start = HAL_GetTick();
    invoke(labels);
    end = HAL_GetTick();
	sprintf(buf,"Elapse time: %d ms\r\n", end - start);
	printLog(buf);
	float *outptr = (float *)getOutput();

	for (int i = 0; i < 200; i++){
		if (i % 4 == 0)
			printLog("\r\n");
		sprintf(buf, "%.4f,  ", *outptr++);
		printLog(buf);
	}
	printLog("\r\nbenchmark ends\r\n");

#elif TESTINT8
	printLog("** INT8 Start **\r\n");
	signed char *input = getInput();
	for (i = 0; i < 3*128*128;i++){
	  input[i] = 1;
	}
	float labels[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	int start_main = HAL_GetTick();
	invoke(labels);

	int end_main = HAL_GetTick();
	sprintf(buf,"Elapse time: %d ms\r\n", end_main - start_main);
	printLog(buf);


	signed char *outptr = getOutput();
	//float* outptr = getOutput_fp();
	//int32_t *outptr = getOutput_int32();
	for (int i = 0; i < 40*64; i++){
		if (i % 40 == 0) {
		  printLog("\r\n");
		}
		//sprintf(buf, "%4d, ", *outptr++);
		//sprintf(buf, "%.6f, ", *outptr++);
		sprintf(buf, "%d, ", *outptr++);
		//sprintf(buf, "%.7e, ", *outptr++);
		printLog(buf);
	}
	printLog("\r\nbenchmark ends\r\n");

#else
  int i;
  for (i = 0; i < INPUT_CH*OUTPUT_CH; i++)
	  w[i] = new_w[i];
  for (i = 0; i < OUTPUT_CH; i++)
	  b[i] = new_b[i];
  uint32_t start, end;
  signed char *input = getInput();
  for (i = 0; i < 3*32*32;i++){
	  input[i] = 1;
  }
  start = HAL_GetTick();

  invoke();

  end = HAL_GetTick();

  sprintf(buf,"Elapse time: %d ms\r\n", end - start);
  printLog(buf);
  signed char *outptr = getOutput();

	for (int i = 0; i < 1 * 1; i++){
		for(int j = 0; j < 10; j++){
			sprintf(buf, "%d,", *outptr++);
			printLog(buf);
		}
	}
  printLog(buf);

#endif
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART6_UART_Init(void)
{

  /* USER CODE BEGIN USART6_Init 0 */

  /* USER CODE END USART6_Init 0 */

  /* USER CODE BEGIN USART6_Init 1 */

  /* USER CODE END USART6_Init 1 */
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 115200;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart6.Init.OverSampling = UART_OVERSAMPLING_16;
  huart6.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart6.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART6_Init 2 */

  /* USER CODE END USART6_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}
