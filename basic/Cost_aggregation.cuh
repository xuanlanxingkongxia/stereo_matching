#pragma once
#include "Post_process.h"
#include <time.h>
#include<opencv2\opencv.hpp>
#include<vector>
#include "cuda_runtime.h"  
#include <stdio.h>
#include <time.h>
#include <iostream>  
#include "device_launch_parameters.h"

#define L1 34
#define L2 17
#define E1 15
#define E2 5

class Cost_aggregation :
	public Post_process
{
public:
	Cost_aggregation(int height_, int width_, int depth_,GpuMat& left_,GpuMat& right_,cudaPitchedPtr gpu_cost);
	~Cost_aggregation();
	void process()override;

private:
	int* limits;
};

enum directions 
{
	l,
	r,
	up,
	down
};