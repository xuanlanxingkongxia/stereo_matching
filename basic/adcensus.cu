#include "adcensus.cuh"
#include<vector>
#include <time.h>
#define block_x 16
#define block_y 16
#define grid_x 78
#define grid_y 24


/*
__global__ void census(PtrStepSz<uchar3> left, PtrStepSz<uchar3> right, cudaPitchedPtr cost, int h, int w, int d, int window)
{
	char* devptr = (char*)cost.ptr;
	size_t pitch = cost.pitch;
	size_t slicepitch = pitch * w;
	for (int q = 0; q < grid_y ; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x + d + window < w && threadid_y < h - window && threadid_x >= window && threadid_y >= window)
		{
			uchar3 colar_l0 = left(threadid_y, threadid_x);
			int* pos = (int*)(devptr + threadid_y * slicepitch + threadid_x * pitch);
			for (int i = 0; i < d; i++)
			{
				uchar3 colar_r0 = right(threadid_y, threadid_x + i);
				pos[i] = 0;
				for (int j = threadid_y - window; j < threadid_y + window; j++)
				{
					for (int k = threadid_x - window; k < threadid_x + window; k++)
					{
						uchar3 colar_l1 = left(j, k);
						uchar3 colar_r1 = right(j, k + i);
						pos[i] += (int)((colar_l1.x > colar_l0.x) != (colar_r1.x > colar_r0.x));
						pos[i] += (int)((colar_l1.y > colar_l0.y) != (colar_r1.y > colar_r0.y));
						pos[i] += (int)((colar_l1.z > colar_l0.z) != (colar_r1.z > colar_r0.z));
					}
				}
			}
		}
	}
}

namespace cuda_cen
{
	void cost_compute(GpuMat& left, GpuMat& right, int* cost, int h, int w, int d, int window)
	{
		cudaPitchedPtr gpu_cost;
		cudaExtent size=make_cudaExtent(d * sizeof(int),w,h);       //the sequence of data stored in cuda is h,w,d(from higher to lower) 
		cudaError_t status;
//		size_t size = h * w * d * sizeof(long);
		status = cudaMalloc3D(&gpu_cost, size);
		if (status != cudaSuccess)
		{
			cout << cudaGetErrorString(status) << endl;
		}
		dim3 blocksize(block_x, block_y);
		dim3 gridsize(grid_x, 1);
		clock_t time0 = clock();
		census << <gridsize, blocksize >> > (left, right, gpu_cost, h, w, d, window);
		status = cudaThreadSynchronize();
		clock_t time1 = clock();
		if (status != cudaSuccess)
		{
			cout << cudaGetErrorString(status) << endl;
		}
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
		cout << "计算耗时:" << time1 - time0 << endl;
		cout << "传输耗时:" << time2 - time1 << endl;
	}
}
*/
