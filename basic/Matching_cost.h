//TODO: add the colar information for my match_cost evaluation
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
#include"device_functions.h"
using namespace cv;
using namespace std;

class Matching_cost
{
public:
	Matching_cost(Mat* left_, Mat* right_, Matx33d R_, Vec3d T_);
	Matching_cost(Mat* left_, Mat* right_);
	~Matching_cost();
	Vec3i map(int i,int j, int d);
	virtual bool evaluate();
	void colared();
	void grayed();
	void set_depth(int depth_);
	int* cost;
	cudaPitchedPtr gpu_cost;

protected:
	Mat* left;
	Mat* right;
	Matx33d R;
	Vec3d T;
	//the sequence of the code is height,width,depth
	int height;
	int width;
	int depth = 256;
	bool if_colar = true;
};

