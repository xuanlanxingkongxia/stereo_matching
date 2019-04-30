#pragma once
#include "Matching_cost.h"
class NCC_cost :
	public Matching_cost
{
public:
	NCC_cost(Mat* left, Mat*right);
	~NCC_cost();
	bool evaluate() override;

protected:
	int patch_size = 11;
	//the sequence of the code is height,width,patch*patch-1
};

