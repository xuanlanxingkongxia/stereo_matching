#include "NCC_cost.h"



NCC_cost::NCC_cost(Mat* left_, Mat* right_)
	:Matching_cost(left_, right_)
{
}


NCC_cost::~NCC_cost()
{
}

bool NCC_cost::evaluate()
{
	
	int margin = (patch_size - 1) / 2;
	for (int i = 0 + margin; i < height - margin; i++)
	{
		for (int j = 0 + margin; j < width - margin; j++)
		{
			for (int k = 0; k < depth; k++)
			{
				if (k < min<int>(depth, width - j))
				{
					cost[i * depth * width + j * depth + k] = 0;
					int  upper=0;
					int divider_l = 0;
					int divider_r = 0;
					for (int p = -margin; p <= margin; p++)
					{
						for (int q = -margin; q <= margin; q++)
						{
							upper += left->at<uchar>(i, j) * right->at<uchar>(i, j + k);
							divider_l += left->at<uchar>(i, j) * left->at<uchar>(i, j + k);
							divider_r += right->at<uchar>(i, j) * right->at<uchar>(i, j + k);
						}
					}
					cost[i * height * width + j * width + k] = 1 - (float)upper*upper / (float)divider_l*divider_r;
				}
				else
				{
					cost[i * depth * width + j * depth + k] = INT_MAX;
				}
			}
		}
	}
	return true;
}
