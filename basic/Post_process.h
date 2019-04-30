#pragma once
#include<vector>
#include<opencv2\opencv.hpp>
#include "cuda_runtime.h"  
#include <stdio.h>
#include <time.h>
#include <iostream>  
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include "cuda.h"
#include"device_functions.h"
using namespace std;
using namespace cv;
using namespace cv::cuda;
#define block_x 64
#define block_y 4
#define grid_x 16
#define grid_y 94


class Post_process
{
public:
	Post_process(int height_, int width_, int depth_, int* cost_);
	Post_process(int height_, int width_, int depth_,GpuMat& left_,GpuMat& right_,cudaPitchedPtr gpu_cost);
	virtual void process();
	cudaPitchedPtr gpu_cost;
	~Post_process();

protected:
	int* cost;
	GpuMat left;
	GpuMat right;
	int h;
	int w;
	int d;
};

