#include "MI_cost.h"

MI_cost::MI_cost(Mat* left_, Mat*right_)
	:Matching_cost(left_, right_)
{
	depth_now.create(height, width, CV_8UC1);
	information.create(256, 256, CV_32SC1);
	srand((unsigned int)time(NULL));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			depth_now.at < uchar >(i, j) = rand() % 256;
		}
	}
	left_whole.resize(256);
	right_whole.resize(256);
}


MI_cost::~MI_cost()
{
	
}

bool MI_cost::evaluate()
{
	for (int t = 0; t < turn; t++)
	{
		compute_information();
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int first = left->at<uchar>(i, j);
				int k;
				for (k = 0; k < min<int>(depth,width - j); k++)
				{
					int second = right->at<uchar>(i, j + k);
					cost[i * depth * width + j * depth + k] = information.at<int>(first, second) - left_whole[first] - right_whole[second];
				}
				for (; k < depth; k++)
				{
					cost[i * depth * width + j * depth + k] = 100000;
				}
			}
		}
		compute_min();
	}
	return true;
}

void MI_cost::compute_min()
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int num = 0;
			long min_ = cost[i * depth * width + j * depth];
			for (int k = 0; k < min<int>(depth, width - k); k++)
			{
				if (cost[i * depth * width + j * depth + k] < min_)
				{
					min_ = cost[i * depth * width + j * depth + k];
					num = k;
				}
			}
			depth_now.at<uchar>(i, j) = num;
		}
	}
}

void MI_cost::compute_information()
{
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			information.at<int>(i, j) = 0;
			left_whole[i] = 0;
			right_whole[i] = 0;
		}
	}
	long sum = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int depth = depth_now.at<uchar>(i, j);
			if (j + depth < width)
			{
				sum += 1;
				int first = left->at<uchar>(i, j);
				int second = right->at<uchar>(i, j + depth);
				information.at<int>(first,second) += 1;
				left_whole[first] = left_whole[first] + 1;
				right_whole[second] = right_whole[second] + 1;
			}
		}
	}
	for (int i = 0; i < 256; i++)
	{
		double medium;
		medium = (double)left_whole[i] / sum;
		left_whole[i] = -(long)log(medium);
		medium = (double)right_whole[i] / sum;
		right_whole[i] = -(long)log(medium);
		for (int j = 0; j < 256; j++)
		{
			medium = (double)information.at<int>(i, j) / sum;
			information.at<int>(i, j) = -(int)log(medium);
		}
	}
}