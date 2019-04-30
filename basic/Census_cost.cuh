//TODO: the way to classify the left and the right is very ugly,i wang to change it in the future
#pragma once
#include "Matching_cost.h"
#include <time.h>
#include<opencv2\opencv.hpp>
#include<vector>
using namespace std;
using namespace cv::cuda;
#define block_x 64
#define block_y 4
#define grid_x 16
#define grid_y 94
#define lamda_ad 10
#define lamda_census 30

class Census_cost :
	public Matching_cost
{
public:
	Census_cost(Mat* left,Mat*right);
	Census_cost(Mat* left_, Mat* right_, Matx33d R_, Vec3d T_);
	~Census_cost();
	bool evaluate() override;
	void cost_compute(GpuMat& left, GpuMat& right, int* cost, int h, int w, int d, int window);


protected:
	int patch_size=9;
	//the sequence of the code is height,width,patch*patch-1
	bool*** code_left;
	bool*** code_right;
//	void encode(int h,int w,Mat* this_left,Mat* this_right);
};

