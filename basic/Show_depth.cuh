#pragma once
#include<opencv2\opencv.hpp>
#include<vector>
#include<iostream>
#include "cuda_runtime.h"  
#include <stdio.h>
#include <time.h>
#include <iostream>  
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include "cuda.h"
#include"device_functions.h"
using namespace cv;
using namespace std;
using namespace cv::cuda;
#define block_x 64
#define block_y 4
#define grid_x 16
#define grid_y 94

class Show_depth
{
public:
	Show_depth(int height_,int width_,int depth_,int* cost_,Mat& depth__,cudaPitchedPtr gpu_cost_);
	~Show_depth();
	void show();
//	void compute_img();
	float error_rate();

protected:
	cudaPitchedPtr gpu_cost;
	Mat img;
	Mat depth;
	int h;
	int w;
	int d;
};

