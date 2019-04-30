#include "Semi_global_matching.cuh"



Semi_global_matching::Semi_global_matching(int height_, int width_, int depth_, int* cost_)
	:Post_process(height_,width_,depth_,cost_)
{
	cost_now.create(h, w, CV_16UC1);
}

Semi_global_matching::Semi_global_matching(int height_, int width_, int depth_, GpuMat& left_, GpuMat& right_, cudaPitchedPtr gpu_cost)
	: Post_process(height_, width_, depth_, left_, right_, gpu_cost)
{
	cost_now.create(h, w, CV_16UC1);
}

Semi_global_matching::~Semi_global_matching()
{
}

void Semi_global_matching::set_image(GpuMat& left_, GpuMat& right_)
{
	left = left_;
	right = right_;
}

__global__ void compute_min(cudaPitchedPtr cost,PtrStepSz<int> depth_min,int h,int w,int d)
{
	char *start = (char*)cost.ptr;
	int pitch = cost.pitch;
	int slicepitch = cost.pitch*w;
	for (int q = 0; q < grid_y; q++)
	{
		int threadid_x = blockIdx.x*blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x < w && threadid_y < h)
		{
			int *pos = (int*)(start + threadid_y * slicepitch + threadid_x * pitch);
			int min_ = pos[0];
			int num = 0;
			for (int i = 0; i < d; i++)
			{
				if (pos[i] < min_)
				{
					min_ = pos[i];
					num = i;
				}
			}
			depth_min(threadid_y, threadid_x) = min_;
		}
	}
}

__global__ void semi_global(cudaPitchedPtr cost, PtrStepSz<uint> cost_now, PtrStepSz<uchar3> left, PtrStepSz<uchar3> right, int h, int w, int d)
{
	char *start = (char*)cost.ptr;
	int pitch = cost.pitch;
	int slicepitch = cost.pitch * w;
	char *middle;
	int *pos;
	int *pos2;
	for (int q = 0; q < grid_y; q++)
	{
		int threadid_x = blockIdx.x * blockDim.x + threadIdx.x;
		int threadid_y = q * blockDim.y + threadIdx.y;
		if (threadid_x + d < w - 1 && threadid_y < h - 1 && threadid_x>=1 && threadid_y>=1)
		{
			middle = start + threadid_y * slicepitch;
			middle = middle + threadid_x * pitch;
			for (int k = 0; k < d; k++)
			{
				pos = (int*)middle;
				for (int x = -1; x <= 1; x++)
				{
					for (int y = -1; y <= 1; y++)
					{
						if (x == 0 && y == 0)
						{
							continue;
						}
						int d1 = abs(left(threadid_y + y, threadid_x + x).z - left(threadid_y, threadid_x).z) + 
							abs(left(threadid_y + y, threadid_x + x).y - left(threadid_y, threadid_x).y) + 
							abs(left(threadid_y + y, threadid_x + x).x - left(threadid_y, threadid_x).x);
						int d2 = abs(right(threadid_y + y, threadid_x + x + d).z - right(threadid_y, threadid_x + d).z) +
							abs(right(threadid_y + y, threadid_x + x + d).y - right(threadid_y, threadid_x + d).y)
							+ abs(right(threadid_y + y, threadid_x + x + d).x - right(threadid_y, threadid_x + d).x);
						int p1;
						int p2;
						if (d1 <= sgm_d)
						{
							if (d2 <= sgm_d)
							{
								p1 = sgm_p1;
								p2 = sgm_p2;
							}
							else
							{
								p1 = sgm_p1 / sgm_q2;
								p2 = sgm_p2 / sgm_q2;
							}
						}
						else
						{
							if (d2 >= sgm_d)
							{
								p1 = sgm_p1 / sgm_q1;
								p2 = sgm_p2 / sgm_q1;
							}
							else
							{
								p1 = sgm_p1 / sgm_q2;
								p2 = sgm_p2 / sgm_q2;
							}
						}
						pos2 = (int*)(middle + (x + threadid_x) * pitch + (y + threadid_y) * slicepitch);
						int t1 = pos2[k];
//						printf("第%d行，第%d列，第%d层 \n", threadid_y, threadid_x, k);
						int a = (k > 1) ? (k - 1) : 0;
						int t2 = pos2[a] + p1;
						int b = k < (d - 1) ? (k + 1) : (d - 1);
						int t3 = pos2[b] + p1;
						int t4 = cost_now(y + threadid_y, x + threadid_x);
						int min1 = t1 < t2 ? t1 : t2;
						int min2 = t3 < t4 ? t3 : t4;
						int min = min1 < min2 ? min1 : min2;
						float t = -cost_now(threadid_y + y, threadid_x + d + x) + min;
						pos[k] += (int)(t / 8);
					}
				}
			}
		}
	}
}

/*
void Semi_global_matching::compute_min()
{
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			int num = 0;
			long min_ = cost[i * d * w + j * d];
			for (int k = 0; k < min<int>(d, w - k); k++)
			{
				if (cost[i * d * w + j * d + k] < min_)
				{
					min_ = cost[i * d * w + j * d + k];
					num = k;
				}
			}
			depth_now.at<float>(i, j) = min_;
		}
	}
}
*/

void Semi_global_matching::process()
{
	dim3 blocksize(block_x, block_y);
	dim3 grid_size(grid_x, 1);
	for (int turn = 0; turn < 10; turn++)
	{
		compute_min << <blocksize, grid_size >> > (gpu_cost, cost_now, h, w, d);
		cudaError_t status;
		status = cudaThreadSynchronize();
		if (status != cudaSuccess)
		{
			cout << cudaGetErrorString(status)<<" 计算最小代价失败"<<endl;
		}
		semi_global << <blocksize, grid_size >> > (gpu_cost, cost_now, left, right, h, w, d);
		status = cudaThreadSynchronize();
		if (status != cudaSuccess)
		{
			cout << cudaGetErrorString(status) << " 优化失败" << endl;
		}
/*		compute_min();
		for (int i = 1; i < h - 1; i++)
		{
			for (int j = 1; j < w - 1; j++)
			{
				int a = *left->ptr<uchar>(i, j);
				int d1 = abs(*left->ptr<uchar>(i, j) - *right->ptr<uchar>(i, j));
				int d2 = abs(*right->ptr<uchar>(i, j) - *right->ptr<uchar>(i, j));
				int p1;
				int p2;
				if (d1 <= sgm_d)
				{
					if (d2 <= sgm_d)
					{
						p1 = sgm_p1;
						p2 = sgm_p2;
					}
					else
					{
						p1 = sgm_p1 / sgm_q2;
						p2 = sgm_p2 / sgm_q2;
					}
				}
				else
				{
					if (d2 >= sgm_d)
					{
						p1 = sgm_p1 / sgm_q1;
						p2 = sgm_p2 / sgm_q1;
					}
					else
					{
						p1 = sgm_p1 / sgm_q2;
						p2 = sgm_p2 / sgm_q2;
					}
				}
				for (int k = 0; k < d; k++)
				{
					for (int x = -1; x <= 1; x++)
					{
						for (int y = -1; y <= 1; y++)
						{
							if (x == 0 && y == 0)
							{
								continue;
							}
							float t = -depth_now.at<float>(i + x, j + y) + min<int>(min<int>(cost[(i + x) * d * w + (j + y) * d + k],
								cost[(i + x) * d * w + (j + y) * d + (k >= 1 ? k - 1 : k)] + p1)
								, min<int>(cost[(i + x) * d * w + (j + y) * d + (k < d - 1 ? k + 1 : k)] + p1, depth_now.at<float>(i + x, j + y) + p2));
							cost[i * d * w + j * d + k] += t/8;
						}
					}*/
/*					float t = -depth_now.at<float>(i + 1, j) + min(min<float>(cost[i + 1][j][k], cost[i + 1][j][k >= 1 ? k - 1 : k] + p1)
						, min<float>(cost[i + 1][j][k < d - 1 ? k + 1 : k] + p1, depth_now.at<float>(i + 1, j) + p2));
					cost[i][j][k] += 0.25* t;
					t = -depth_now.at<float>(i - 1, j ) + min(min<float>(cost[i + 1][j][k], cost[i - 1][j ][k >= 1 ? k - 1 : k] + p1)
						, min<float>(cost[i - 1][j][k < d - 1 ? k + 1 : k] + p1, depth_now.at<float>(i - 1, j) + p2));
					cost[i][j][k] += 0.25* t;
					t = -depth_now.at<float>(i, j + 1) + min(min<float>(cost[i][j + 1][k], cost[i][j + 1][k >= 1 ? k - 1 : k] + p1)
						, min<float>(cost[i][j + 1][k < d - 1 ? k + 1 : k] + p1, depth_now.at<float>(i, j + 1) + p2));
					cost[i][j][k] += 0.25* t;
					t = -depth_now.at<float>(i - 1, j - 1) + min(min<float>(cost[i - 1][j - 1][k], cost[i - 1][j - 1][k >= 1 ? k - 1 : k] + p1)
						, min<float>(cost[i - 1][j - 1][k < d - 1 ? k + 1 : k] + p1, depth_now.at<float>(i - 1, j - 1) + p2));
					cost[i][j][k] += 0.25* t;
				}
			}
		}*/
	}
}