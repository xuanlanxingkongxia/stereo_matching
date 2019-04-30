#include "Cost_aggregation.cuh"

Cost_aggregation::Cost_aggregation(int height_, int width_, int depth_, GpuMat& left_,GpuMat& right_,cudaPitchedPtr gpu_cost)
	:Post_process(height_, width_, depth_, left_, right_,gpu_cost)
{
}


Cost_aggregation::~Cost_aggregation()
{
}

__global__ void initialize_num(int h, int w, int d, int*num)
{
	for (int q = 0; q < grid_y; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x < w && threadid_y < h  && threadid_x >= 0 && threadid_y >= 0)
		{
			for (int i = 0; i < d; i++)
			{
				num[threadid_y * w * d + threadid_x * d + i] = 1;
			}
		}
	}
}

__global__ void normlize(cudaPitchedPtr cost, int h, int w, int d, int* num)
{
	char* devptr = (char*)cost.ptr;
	size_t pitch = cost.pitch;
	size_t slicepitch = pitch * w;
	for (int q = 0; q < grid_y; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		int* pos = (int*)(devptr + threadid_y * slicepitch + threadid_x * pitch);
		if (threadid_x < w && threadid_y < h  && threadid_x >= 0 && threadid_y >= 0)
		{
			for (int i = 0; i < d; i++)
			{
				pos[i] /= (float)num[threadid_y * w * d + threadid_x * d + i];
			}
		}
	}
}


__global__ void aggregation(cudaPitchedPtr cost, cudaPitchedPtr cost2, int h, int w, int d, int* limits,int* num, int* num2,bool direction_h)
{
	int* devptr = (int*)cost.ptr;
	int* devptr2 = (int*)cost2.ptr;
	size_t pitch = cost.pitch / sizeof(int);
	size_t slicepitch = pitch * w;
	for (int q = 0; q < grid_y; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x < w && threadid_y < h  && threadid_x >= 0 && threadid_y >= 0)
		{
			for (int i = 0; i < d; i++)
			{
				int sum_num = 0;
				int sum = 0;
				if (direction_h)
				{
					int plus = limits[up * h * w + threadid_y * w + threadid_x];
					int minus = -limits[down * h * w + threadid_y * w + threadid_x];
					for (int k = minus; k <= plus; k++)
					{
						sum = sum + devptr[(threadid_y + k) * w * pitch + threadid_x * pitch + i];
						sum_num = sum_num + num[(threadid_y + k) * w * pitch + threadid_x * pitch + i];
/*						if (threadid_x == 500 && threadid_y == 150 && i == 10)
						{
							printf("第%d轮：", k);
							printf("sum:%d \n", devptr[threadid_y * w * pitch + threadid_x * pitch + i]);
							printf("sum_origin:%d \n", sum);
							printf("sum_num:%d \n", num[threadid_y * w * pitch + threadid_x * pitch + i]);
							printf("sum_num_origin:%d \n \n", sum_num);
						}*/
					}
				}
				else
				{
					int plus = limits[r * h * w + threadid_y * w + threadid_x];
					int minus = -limits[l * h * w + threadid_y * w + threadid_x];
					for (int k = minus; k <= plus; k++)
					{
						sum = sum + devptr[threadid_y * w * pitch +(threadid_x + k ) * pitch + i];
						sum_num = sum_num + num[threadid_y * w * pitch + (threadid_x + k) * pitch + i];
					}
				}
				devptr2[threadid_y * w * pitch + threadid_x * pitch + i] = sum;
				num2[threadid_y * w * pitch + threadid_x * pitch + i] = sum_num;
/*				if (threadid_x == 500 && threadid_y == 150 && i==10)
				{
					printf("sum:%d \n", devptr[threadid_y * w * pitch + threadid_x * pitch + i]);
					printf("sum_origin:%d \n", sum);
					printf("sum_num:%d \n", num[threadid_y * w * pitch + threadid_x * pitch + i]);
					printf("sum_num_origin:%d \n \n", sum_num);
				}*/
			}
		}
	}
}

__global__ void compute_limits(PtrStepSz<uchar3> left, int h, int w, int dir_x, int dir_y, int* distance)
{
	for (int q = 0; q < grid_y; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x < w && threadid_y < h  && threadid_x >= 0 && threadid_y >= 0)
		{
			uchar3 colar0 = left(threadid_y, threadid_x);
			int i = 1;
			for (; i < L1; i++)
			{
				if (i * dir_x + threadid_x >= w || i * dir_y + threadid_y >= h || i * dir_x + threadid_x < 0 || i * dir_y + threadid_y < 0)
				{
					break;
				}
				uchar3 colar1 = left(threadid_y + i * dir_y, threadid_x + i * dir_x);
				int diff_x = abs(colar1.x - colar0.x);
				int diff_y = abs(colar1.y - colar0.y);
				int diff_z = abs(colar1.z - colar0.z);
				if (diff_x > E1 || diff_y > E1 || diff_z > E1)
				{
					break;
				}
			}
			for (; i >= L1 && i < L2; i++)
			{
				if (i * dir_x + threadid_x >= w || i * dir_y + threadid_y >= h || i * dir_x + threadid_x < 0 || i * dir_y + threadid_y < 0)
				{
					break;
				}
				uchar3 colar1 = left(threadid_y + i * dir_y, threadid_x + i * dir_x);
				int diff_x = abs(colar1.x - colar0.x);
				int diff_y = abs(colar1.y - colar0.y);
				int diff_z = abs(colar1.z - colar0.z);
				if (diff_x > E2 || diff_y > E2 || diff_z > E2)
				{
					break;
				}
			}
			distance[threadid_y*w + threadid_x] = i - 1;
		}
	}
}



void Cost_aggregation::process()
{
	dim3 blocksize(block_x,block_y);
	dim3 gridsize(grid_x,1);
	cudaError_t status;
	size_t size = h * w * sizeof(int);
	cudaMalloc((void**)&(limits), 4 * size);
	compute_limits << <gridsize, blocksize >> > (left, h, w, 1, 0, &limits[r*h*w]);
	cudaThreadSynchronize();
	compute_limits << <gridsize, blocksize >> > (left, h, w, -1, 0, &limits[l*h*w]);
	cudaThreadSynchronize();
	compute_limits << <gridsize, blocksize >> > (left, h, w, 0, 1, &limits[up*h*w]);
	cudaThreadSynchronize();
	compute_limits << <gridsize, blocksize >> > (left, h, w, 0, -1, &limits[down*h*w]);
	status = cudaThreadSynchronize();
	cudaPitchedPtr gpu_cost2;
	cudaExtent cost_size = make_cudaExtent(d * sizeof(int), w, h);
	cudaMalloc3D(&gpu_cost2, cost_size);
	int* num;
	int* num2;
	size_t size2 = h * w * d * sizeof(int);
	cudaMalloc((void**)&num, size2);
	cudaMalloc((void**)&num2, size2);
	initialize_num<<<gridsize,blocksize>>>(h, w, d, num);
	cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << "计算限制部分";
	}
	aggregation << <gridsize, blocksize >> > (gpu_cost, gpu_cost2, h, w, d, limits, num, num2, true);
	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << "计算聚合部分";
	}
	aggregation << <gridsize, blocksize >> > (gpu_cost2, gpu_cost, h, w, d, limits, num2, num, false);
	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << "计算聚合部分";
	}
	normlize << <gridsize, blocksize >> > (gpu_cost, h, w, d, num);
	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << "计算聚合部分";
	}
	initialize_num << <gridsize, blocksize >> > (h, w, d, num);
	cudaThreadSynchronize();
	aggregation << <gridsize, blocksize >> > (gpu_cost, gpu_cost2, h, w, d, limits, num, num2, false);
	status = cudaThreadSynchronize();
	aggregation << <gridsize, blocksize >> > (gpu_cost2, gpu_cost, h, w, d, limits, num2, num, true);
	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status) << "计算聚合部分";
	}
	normlize << <gridsize, blocksize >> > (gpu_cost, h, w, d, num);
	cudaThreadSynchronize();

	cudaFree(limits);
	cudaFree(num);
	cudaFree(num2);
}

