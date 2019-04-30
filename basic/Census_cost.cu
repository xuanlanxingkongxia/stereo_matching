#pragma once
#include "Census_cost.cuh"


Census_cost::Census_cost(Mat* left_,Mat* right_)
	:Matching_cost(left_,right_)
{
/*	code_left = new bool**[height];
	code_right = new bool**[height];
	for (int i = 0; i < height; i++)
	{
		code_left[i] = new bool*[width];
		code_right[i] = new bool*[width];
		for (int j = 0; j < width; j++)
		{
			code_left[i][j] = new bool[patch_size*patch_size];
			code_right[i][j] = new bool[patch_size*patch_size];
		}
	}*/
}

Census_cost::Census_cost(Mat* left_, Mat* right_,Matx33d R_,Vec3d T_)
	:Matching_cost(left_, right_, R_, T_)
{
	code_left = new bool**[height];
	code_right = new bool**[height];
	for (int i = 0; i < height; i++)
	{
		code_left[i] = new bool*[width];
		code_right[i] = new bool*[width];
		for (int j = 0; j < width; j++)
		{
			code_left[i][j] = new bool[patch_size*patch_size];
			code_right[i][j] = new bool[patch_size*patch_size];
		}
	}
}

Census_cost::~Census_cost()
{
/*	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			delete[] code_left[i][j];
			delete[] code_right[i][j];
		}
		delete[] code_left[i];
		delete[] code_right[i];
	}
	delete[] code_left;
	delete[] code_right;*/
}

bool Census_cost::evaluate()
{
/*	int test=ad();
	int margin = (patch_size - 1) / 2;
	for (int i = 0 + margin; i < height - margin; i++)
	{
		for (int j = 0 + margin; j < width - margin; j++)
		{
			encode(i, j, left,right);
		}
	}
	Vec3i medium;
	for (int i = 0 + margin; i < height - margin; i++)
	{
		for (int j = 0 + margin; j < width - margin; j++)
		{
			medium = map(i, j, 255);
			if (medium[0] < width && medium[1] < height && medium[1]>0 && medium[0]>0)
			{
				for (int k = 0; k < depth; k++)
				{
					medium = map(i, j, k);
					if (k < depth && medium[0] < width && medium[1] < height)
					{
						cost[i][j][k] = 0;
						for (int q = 0; q < patch_size*patch_size - 1; q++)
						{
							if (code_left[i][j][q] != code_right[medium[1]][medium[0]][q])
							{
								cost[i][j][k]++;
							}
						}
					}
				}
			}
			else
			{
				cost[i][j][0] = LONG_MIN;
			}
		}
	}
	return true;*/

	setDevice(0);
	if (getCudaEnabledDeviceCount() == 0) {
		cerr << "此OpenCV编译的时候没有启用CUDA模块" << endl;
		return -1;
	}
	GpuMat left_g(left->rows,left->cols,CV_8UC3);
	left_g.upload(*left);
	GpuMat right_g(right->rows, right->cols, CV_8UC3);
	right_g.upload(*right);
	clock_t start = clock();
	cost_compute(left_g, right_g, cost, height, width, depth, (patch_size - 1) / 2);
	clock_t end = clock();
	cout << "gpu运算时间:" <<end - start << endl;
	return true;
}
/*
void Census_cost::encode(int h,int w,Mat* this_left,Mat* this_right)
{
	int margin = (patch_size - 1) / 2;
	for (int i = -margin; i <= margin; i++)
	{
		for (int j = -margin; j <= margin; j++)
		{
			if (*this_left->ptr<uchar>(h + i, w + j) > *this_left->ptr<uchar>(h, w))
			{
				code_left[h][w][(i + margin)*patch_size + j + margin] = true;
			}
			else
			{
				code_left[h][w][(i + margin)*patch_size + j + margin] = false;
			}
			if (*this_right->ptr<uchar>(h + i, w + j) > *this_right->ptr<uchar>(h, w))
			{
				code_right[h][w][(i + margin)*patch_size + j + margin] = true;
			}
			else
			{
				code_right[h][w][(i + margin)*patch_size + j + margin] = false;
			}
		}
	}
	return;
}
*/

__global__ void census(PtrStepSz<uchar3> left, PtrStepSz<uchar3> right, cudaPitchedPtr cost, int h, int w, int d, int window,int standard)
{
	char* devptr = (char*)cost.ptr;
	size_t pitch = cost.pitch;
	size_t slicepitch = pitch * w;
	for (int q = 0; q < grid_y ; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x + d * standard + window < w && threadid_y < h - window && threadid_x >= window && threadid_y >= window)
		{
			uchar3 colar_l0 = left(threadid_y, threadid_x);
			int* pos = (int*)(devptr + threadid_y * slicepitch + threadid_x * pitch);
			for (int i = 0; i < d; i++)
			{
				uchar3 colar_r0 = right(threadid_y, threadid_x + (int)(i*standard));
				pos[i] = 0;
				float a = 0;
				for (int j = threadid_y - window; j < threadid_y + window; j++)
				{
					for (int k = threadid_x - window; k < threadid_x + window; k++)
					{
						uchar3 colar_l1 = left(j, k);
						uchar3 colar_r1 = right(j, k + (int)(i * standard));
						a += ((colar_l1.x > colar_l0.x) != (colar_r1.x > colar_r0.x));
						a += ((colar_l1.y > colar_l0.y) != (colar_r1.y > colar_r0.y));
						a += ((colar_l1.z > colar_l0.z) != (colar_r1.z > colar_r0.z));
					}
				}
				float ad = abs(colar_l0.x - colar_r0.x) + abs(colar_l0.y - colar_r0.y) + abs(colar_l0.z - colar_r0.z);
//				ad = ad;
				pos[i] = 100 - 100*exp(-a / lamda_census) + 100 - 100 * exp(-ad/lamda_ad);
			}
		}
	}
}

void Census_cost::cost_compute(GpuMat& left, GpuMat& right, int* cost, int h, int w, int d, int window)
{
	cudaExtent size=make_cudaExtent(d * sizeof(int),w,h);       //the sequence of data stored in cuda is h,w,d(from higher to lower) 
	cudaError_t status;
//	size_t size = h * w * d * sizeof(long);
	status = cudaMalloc3D(&gpu_cost, size);
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << endl;
	}
	dim3 blocksize(block_x, block_y);
	dim3 gridsize(grid_x, 1);
	clock_t time0 = clock();
	census << <gridsize, blocksize >> > (left, right, gpu_cost, h, w, d, window, T[0]);
	status = cudaThreadSynchronize();
	clock_t time1 = clock();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << endl;
	}
	/*
	cudaMemcpy3DParms pos = { 0 };
	pos.srcPtr = gpu_cost;
	pos.kind = cudaMemcpyDeviceToHost;
	pos.extent = make_cudaExtent(d * sizeof(int), w, h);
	pos.dstPtr = make_cudaPitchedPtr((void*)cost, d * sizeof(int), d, w);
	status = cudaMemcpy3D(&pos);
	clock_t time2 = clock();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << endl;
	}
	*/
	cout << "计算耗时:" << time1 - time0 << endl;
//	cout << "传输耗时:" << time2 - time1 << endl;
}