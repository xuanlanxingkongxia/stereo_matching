#pragma once
#include "Matching_cost.h"
#define turn 10
class MI_cost :
	public Matching_cost
{
public:
	MI_cost(Mat* left_, Mat*right_);
	~MI_cost();
	bool evaluate() override;

protected:
	int patch_size=100;
	Mat information;
	Mat depth_now;
	vector<long> left_whole;
	vector<long> right_whole;
	void compute_min();
	void compute_information();
};