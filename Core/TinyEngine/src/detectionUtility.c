/*
 * detectionUtility.cpp
 *
 *  Created on: Jan 26, 2021
 *      Author: wmchen
 */

#include "math.h"
#include "string.h"
#include "detectionUtility.h"
#define PERSON

/* network_output[:, :, x, :], functions: x is the d1's dimension */
void sigmoid_(float* array, int d0, int d1, int d2, int x){
	int i, cnt_d1;
	for(cnt_d1 = 0; cnt_d1 < d2; cnt_d1++){
		float* start_ptr = &array[d0 * (d1* cnt_d1 + x)];
		for(i = 0; i < d0; i++){
			start_ptr[i] = (1 / (1 + exp(-1.0 * start_ptr[i])));
		}
	}
}

void add_(float* array, int d0, int d1, int d2, int x, float *add_array){
	int i, cnt_d1;
	for(cnt_d1 = 0; cnt_d1 < d2; cnt_d1++){
		float* start_ptr = &array[d0 * (d1* cnt_d1 + x)];
		for(i = 0; i < d0; i++){
			start_ptr[i] = start_ptr[i] + add_array[i];
		}
	}
}

void div_(float* array, int d0, int d1, int d2, int x, float divider){
	int i, cnt_d1;
	for(cnt_d1 = 0; cnt_d1 < d1; cnt_d1++){
		float* start_ptr = &array[d0 * (d1* cnt_d1 + x)];
		for(i = 0; i < d0; i++){
			start_ptr[i] = start_ptr[i] / divider;
		}
	}
}

void exp_(float* array, int d0, int d1, int d2, int x){
	int i, cnt_d1;
	for(cnt_d1 = 0; cnt_d1 < d2; cnt_d1++){
		float* start_ptr = &array[d0 * (d1* cnt_d1 + x)];
		for(i = 0; i < d0; i++){
			start_ptr[i] = exp(start_ptr[i]);
		}
	}
}

void mul_(float* array, int d0, int d1, int d2, int x, float *mul_array){
	int i, cnt_d1;
	for(cnt_d1 = 0; cnt_d1 < d2; cnt_d1++){
		float* start_ptr = &array[d0 * (d1* cnt_d1 + x)];
		for(i = 0; i < d0; i++){
			start_ptr[i] = start_ptr[i] * mul_array[cnt_d1];
		}
	}
}

void softmax_dim2(float* array, int d0, int d1, int d2, int x){
	int i, cnt_d1;
	for(cnt_d1 = 0; cnt_d1 < d2; cnt_d1++){
		float* start_ptr = &array[d0 * (d1* cnt_d1 + x)];
		for(i = 0; i < d0; i++){
#ifdef FACE
			float sum = exp(start_ptr[i]) + exp(start_ptr[i + d0]);
			start_ptr[i] = exp(start_ptr[i]) / sum;
			start_ptr[i + d0] = exp(start_ptr[i + d0]) / sum;
#endif
#ifdef PERSON
			float sum = exp(start_ptr[i]);
			start_ptr[i] = exp(start_ptr[i]) / sum;
#endif
		}
	}
}

void unsqueeze_expand2_mul_reorg(float* array, int d0, int d1, int d2, unsigned char* runtime_buffer){
	int i, j;
	float *cls_scores = (float*)&runtime_buffer[0];//[5*50]
	for(i = 0; i < d2; i++){
		for(j = 0; j < d0; j++){
#ifdef FACE
			cls_scores[i * d0 * 2 + j*2 + 0] = array[i * d0 * d1 + 4 * d0 + j] * array[i * d0 * d1 + 5 * d0 + j];
			cls_scores[i * d0 * 2 + j*2 + 1] = array[i * d0 * d1 + 4 * d0 + j] * array[i * d0 * d1 + 6 * d0 + j];
#endif
#ifdef PERSON
			cls_scores[i * d0 + j] = array[i * d0 * d1 + 4 * d0 + j] * array[i * d0 * d1 + 5 * d0 + j];
#endif
		}
	}
}
/* end: network_output[:, :, x, :] */

float max(float a, float b){
	if(a > b)
		return a;
	else
		return b;
}

float min(float a, float b){
	if(a > b)
		return b;
	else
		return a;
}

#ifdef FACE
	#define NUM_ANCHORS 5
	float anchors_x[5] = {4, 10, 19, 35, 72}, anchors_y[5] = {7, 16, 31, 55, 96};
#endif
#ifdef PERSON
//	#define NUM_ANCHORS 5
//	float anchors_x[5] = {3, 8, 15, 34, 88}, anchors_y[5] = {7, 18, 41, 76, 115};
	#define NUM_ANCHORS 4
	float anchors_x[4] = {8, 15, 34, 88}, anchors_y[4] = {18, 41, 76, 115};
#endif
int postProcessing(signed char *input, unsigned char* runtime_buffer,
		int y_zero, float y_scale, int shape_x, int shape_y, int shape_c,
		int resolution, int width, int height, float conf_thresh, float out_boxes[10][6]){
	int i, j, h, w, c;
	unsigned char* origin = runtime_buffer;
	float* float_output = (float*)runtime_buffer;
	//HWC to CHW: consistent with the training side
	for(c = 0; c < shape_c; c++){
		for(h = 0; h < shape_y; h++){
			for(w = 0; w < shape_x; w++){
				float_output[c * shape_x * shape_y + h * shape_x + w] = (input[c + w * shape_c + h * shape_c * shape_x]- y_zero) * y_scale;
			}
		}
	}
	runtime_buffer += sizeof(float) * shape_c * shape_y * shape_x;

	float reduction = resolution * (1.0 / shape_x);

	//Compute xc, yc, w,h, box_score on Tensor
	float *lin_x = (float*)runtime_buffer;
	runtime_buffer += sizeof(float) * shape_x * shape_y;//[5][50]
	float *lin_y = (float*)runtime_buffer;
	runtime_buffer -= sizeof(float) * shape_x * shape_y;//RECYCLE THIS AFTERWARD
	for(i = 0; i < shape_y; i ++){
		for(j = 0; j < shape_x;j++){
			lin_x[i * shape_x + j] = j;
			lin_y[i * shape_x + j] = i;
		}
	}
	float anchor_w[NUM_ANCHORS], anchor_h[NUM_ANCHORS];
	for(i = 0; i < NUM_ANCHORS;i++){
		anchor_w[i] = anchors_x[i] / reduction;
	}
	for(i = 0; i < NUM_ANCHORS;i++){
		anchor_h[i] = anchors_y[i] / reduction;
	}

	//network_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)           # X center
	sigmoid_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 0);
	add_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 0, lin_x);
	div_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 0, (float) shape_x);
	//network_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)           # Y center
	sigmoid_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 1);
	add_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 1, lin_y);
	div_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 1, (float) shape_y);
	//network_output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)            # Width
	exp_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 2);
	mul_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 2, anchor_w);
	div_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 2, (float) shape_x);
	//network_output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)            # Height
	exp_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 3);
	mul_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 3, anchor_h);
	div_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 3, (float) shape_y);
	//network_output[:, :, 4, :].sigmoid_()                               # Box score
	sigmoid_(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 4);

	//# Compute class_score
    //cls_scores = torch.nn.functional.softmax(network_output[:, :, 5:, :], 2)
	softmax_dim2(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, 5);
	//cls_scores = (cls_scores * conf_scores.unsqueeze(2).expand_as(cls_scores)).transpose(2,3)
	//cls_scores = cls_scores.contiguous().view(cls_scores.size(0), cls_scores.size(1), -1)
//	float cls_scores[5 * shape_y * shape_x * 2];
	float *cls_scores = (float *)runtime_buffer;
	unsqueeze_expand2_mul_reorg(float_output, shape_x * shape_y, shape_c/NUM_ANCHORS, NUM_ANCHORS, runtime_buffer);
	runtime_buffer += sizeof(float) * 5 * shape_y * shape_x * 2;//[5][50]

	//score_thresh = cls_scores > conf_thresh
	//coords = network_output.transpose(2, 3)[..., 0:4]
    //coords = coords.unsqueeze(3).expand(coords.size(0),coords.size(1),coords.size(2),num_classes,coords.size(3)).contiguous().view(coords.size(0),coords.size(1),-1,coords.size(3))
	//coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
    //idx = (torch.arange(num_classes)).repeat(batch, num_anchors, w*h)# .cuda()
    //idx = idx[score_thresh].view(-1, 1).float()
    //detections = torch.cat([coords, scores, idx], dim=1)
#define MAXBOX 40
	float *detections = (float *)runtime_buffer;
	runtime_buffer += sizeof(float) * 5 * MAXBOX * 2 * 6;//[5*50][6]
	float areas[5 * MAXBOX * 2];
	runtime_buffer += sizeof(float) * 5 * MAXBOX * 2;//[5*50]
	unsigned char suppress[5 * MAXBOX * 2];
	runtime_buffer += sizeof(unsigned char ) * 5 * MAXBOX * 2;//[5*50]
	memset(suppress, 1, 5 * MAXBOX * 2);
	int num_boxs = 0;

	for(i = 0; i < 5; i++){
#ifdef PERSON
		for(j = 0; j < shape_y * shape_x; j++){
			if(num_boxs >= MAXBOX)
				break;
			if(cls_scores[i * shape_y * shape_x + j] > conf_thresh){
				//boxes = detections
				//boxes[:, 0:3:2].mul_(width)
				//boxes[:, 1:4:2].mul_(height)
				detections[num_boxs * 6 + 0] = float_output[(i * shape_c/NUM_ANCHORS + 0) * shape_x * shape_y + j] * (float)(width);
				detections[num_boxs * 6 + 1] = float_output[(i * shape_c/NUM_ANCHORS + 1) * shape_x * shape_y + j] * (float)(height);
				detections[num_boxs * 6 + 2] = float_output[(i * shape_c/NUM_ANCHORS + 2) * shape_x * shape_y + j] * (float)(width);
				detections[num_boxs * 6 + 3] = float_output[(i * shape_c/NUM_ANCHORS + 3) * shape_x * shape_y + j] * (float)(height);
				detections[num_boxs * 6 + 4] = cls_scores[i * shape_y * shape_x + j];
				detections[num_boxs * 6 + 5] = 0;

				//boxes[:, 0] -= boxes[:, 2] / 2
				//boxes[:, 1] -= boxes[:, 3] / 2
				//boxes[:, 2] += boxes[:, 0]
				//boxes[:, 3] += boxes[:, 1]
				detections[num_boxs * 6 + 0] -= detections[num_boxs * 6 + 2] / 2.0;
				detections[num_boxs * 6 + 1] -= detections[num_boxs * 6 + 3] / 2.0;
				detections[num_boxs * 6 + 2] += detections[num_boxs * 6 + 0];
				detections[num_boxs * 6 + 3] += detections[num_boxs * 6 + 1];

				//x1 = dets[:, 0]
				//y1 = dets[:, 1]
				//x2 = dets[:, 2]
				//y2 = dets[:, 3]
				//scores = dets[:, 4]
				//areas = (x2 - x1 + 1) * (y2 - y1 + 1)
				areas[num_boxs] = (detections[num_boxs * 6 + 2] - detections[num_boxs * 6 + 0] + 1) * (detections[num_boxs * 6 + 3] - detections[num_boxs * 6 + 1] + 1);

				suppress[num_boxs] = 0;
				num_boxs++;
			}
		}
#endif
#ifdef FACE
		for(j = 0; j < shape_y * shape_x * 2; j++){
			if(cls_scores[i * shape_y * shape_x + j] > conf_thresh){
				//boxes = detections
				//boxes[:, 0:3:2].mul_(width)
				//boxes[:, 1:4:2].mul_(height)
				detections[num_boxs * 6 + 0] = float_output[(i * shape_c/NUM_ANCHORS + 0) * shape_x * shape_y  + j / 2] * (float)(width);
				detections[num_boxs * 6 + 1] = float_output[(i * shape_c/NUM_ANCHORS + 1) * shape_x * shape_y  + j / 2] * (float)(height);
				detections[num_boxs * 6 + 2] = float_output[(i * shape_c/NUM_ANCHORS + 2) * shape_x * shape_y  + j / 2] * (float)(width);
				detections[num_boxs * 6 + 3] = float_output[(i * shape_c/NUM_ANCHORS + 3) * shape_x * shape_y  + j / 2] * (float)(height);
				detections[num_boxs * 6 + 4] = cls_scores[i * shape_y * shape_x * 2 + j];
				detections[num_boxs * 6 + 5] = j % 2;

				//boxes[:, 0] -= boxes[:, 2] / 2
				//boxes[:, 1] -= boxes[:, 3] / 2
				//boxes[:, 2] += boxes[:, 0]
				//boxes[:, 3] += boxes[:, 1]
				detections[num_boxs * 6 + 0] -= detections[num_boxs * 6 + 2] / 2.0;
				detections[num_boxs * 6 + 1] -= detections[num_boxs * 6 + 3] / 2.0;
				detections[num_boxs * 6 + 2] += detections[num_boxs * 6 + 0];
				detections[num_boxs * 6 + 3] += detections[num_boxs * 6 + 1];

				//x1 = dets[:, 0]
				//y1 = dets[:, 1]
				//x2 = dets[:, 2]
				//y2 = dets[:, 3]
				//scores = dets[:, 4]
				//areas = (x2 - x1 + 1) * (y2 - y1 + 1)
				areas[num_boxs] = (detections[num_boxs * 6 + 2] - detections[num_boxs * 6 + 0] + 1) * (detections[num_boxs * 6 + 3] - detections[num_boxs * 6 + 1] + 1);

				suppress[num_boxs] = 0;
				num_boxs++;
			}
		}
#endif
	}

	//nms
	#define NMS_thresh 0.3
	int max_index = 0, nms_boxes = 0;
	while(1){
		//find the maximum
		max_index = -1;
		float max_score = 0;
		for(i = 0; i < num_boxs; i++){
			if(!suppress[i]){
				if(detections[i * 6 + 4] > max_score){
					max_score = detections[i * 6 + 4];
					max_index = i;
				}
			}
		}
		if(max_index == -1)//cannot find any box
			break;
		else{//add that box
			out_boxes[nms_boxes][0] = detections[max_index * 6 + 0];
			out_boxes[nms_boxes][1] = detections[max_index * 6 + 1];
			out_boxes[nms_boxes][2] = detections[max_index * 6 + 2];
			out_boxes[nms_boxes][3] = detections[max_index * 6 + 3];
			out_boxes[nms_boxes][4] = detections[max_index * 6 + 4];
			out_boxes[nms_boxes][5] = detections[max_index * 6 + 5];
			nms_boxes++;
			suppress[max_index] = 1;//suppress since we add it to the output boxes
		}

		//get the box
		float x1 = detections[max_index * 6 + 0], y1 = detections[max_index * 6 + 1], x2 = detections[max_index * 6 + 2], y2 = detections[max_index * 6 + 3];
		for(i = 0; i < num_boxs; i++){
			if(!suppress[i]){
				float xx1 = max(x1, detections[i * 6 + 0]);
				float yy1 = max(y1, detections[i * 6 + 1]);
				float xx2 = min(x2, detections[i * 6 + 2]);
				float yy2 = min(y2, detections[i * 6 + 3]);

				float b_w = max(0.0f, xx2 - xx1 + 1);
				float b_h = max(0.0f, yy2 - yy1 + 1);

				float inter = b_w * b_h;

				float ovr = inter / (areas[max_index] + areas[i] - inter);

				//filter highly-overlapped boxes
				if(ovr > NMS_thresh)
					suppress[i] = 1;
			}
		}
	}

	return nms_boxes;
}




