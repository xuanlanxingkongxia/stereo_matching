#include "Matching_cost.h"


Matching_cost::Matching_cost(Mat* left_,Mat* right_,Matx33d R_,Vec3d T_)
{
	left = left_;
	right = right_;
	R = R_;
	T = T_;
	height = left->rows;
	width = left->cols;
	cost = new int[height*width*depth];
	if (if_colar = true)
	{
	}
	else
	{
		cvtColor(*left, *left, CV_RGB2GRAY);
		cvtColor(*right, *right, CV_RGB2GRAY);
	}
}

Matching_cost::Matching_cost(Mat* left_, Mat* right_)
{
	left = left_;
	right = right_;
	R = Matx33d::eye();
	T[0] = 1;
	T[1] = 0;
	T[2] = 0;
	height = left->rows;
	width = left->cols;
	cost = new int[height*width*depth];
	if (if_colar = true)
	{
	}
	else
	{
		cvtColor(*left, *left, CV_RGB2GRAY);
		cvtColor(*right, *right, CV_RGB2GRAY);
	}
}

Matching_cost::~Matching_cost()
{
	delete cost;
}

Vec3i Matching_cost::map(int i,int j, int d)
{
	Vec3d sourse(j, i, 1);
	sourse = R * sourse + d * T;
	for (int i = 0; i < 3; i++)
	{
		sourse[i] = sourse[i] / sourse[2];
	}
	return sourse;
}

bool Matching_cost::evaluate()
{
	return true;
}

void Matching_cost::set_depth(int depth_)
{
	depth = depth_;
	return;
}

void Matching_cost::colared()
{
	if_colar = true;
}

void Matching_cost::grayed()
{
	if_colar = false;
}