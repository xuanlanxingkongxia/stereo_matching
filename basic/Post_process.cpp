#include "Post_process.h"



Post_process::Post_process(int height_, int width_, int depth_, int* cost_)
{
	cost = cost_;
	h = height_;
	w = width_;
	d = depth_;
}

Post_process::Post_process(int height_, int width_, int depth_,GpuMat& left_,GpuMat& right_,cudaPitchedPtr gpu_cost_)
{
	h = height_;
	w = width_;
	d = depth_;
	left = left_;
	right = right_;
	gpu_cost = gpu_cost_;
}

Post_process::~Post_process()
{
}

void Post_process::process()
{
	return;
}