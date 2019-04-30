#pragma once
#include<opencv2\opencv.hpp>
#include<vector>
#include<time.h>
#include "cuda_runtime.h"  
#include <stdio.h>
#include <time.h>
#include <iostream>  
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include "cuda.h"
using namespace std;
using namespace cv::cuda;

namespace cuda_cen
{
	//void cost_compute(GpuMat& left, GpuMat& right, int* cost, int h, int w, int d, int window);
}