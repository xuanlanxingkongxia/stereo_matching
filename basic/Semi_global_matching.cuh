#pragma once
#include "Post_process.h"
#define sgm_p1 2
#define sgm_p2 54
#define sgm_q1 3
#define sgm_q2 6
#define sgm_d 41

class Semi_global_matching :
	public Post_process
{
public:
	Semi_global_matching(int height_, int width_, int depth_, int* cost_);
	Semi_global_matching(int height_, int width_, int depth_, GpuMat& left_, GpuMat& right_, cudaPitchedPtr gpu_cost);
	~Semi_global_matching();
	void process() override;
	void set_image(GpuMat& left_,GpuMat& right_);
	cudaPitchedPtr cost;
//	void compute_min();

protected:
	
	GpuMat cost_now;
};