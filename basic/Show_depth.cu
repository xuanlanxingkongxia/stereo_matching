#include "Show_depth.cuh"

Show_depth::Show_depth(int height_, int width_, int depth_,int* cost_,Mat& depth__,cudaPitchedPtr gpu_cost_)
{
	h = height_;
	w = width_;
	d = depth_;
	depth = depth__;
	gpu_cost = gpu_cost_;
}


Show_depth::~Show_depth()
{
}

__global__ void gpu_add(cudaPitchedPtr cost, PtrStepSz<uchar1> img,int w, int h)
{
	int threadid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int threadid_y = blockIdx.y*blockDim.y + threadIdx.y;
	if (threadid_x < w && threadid_y < h && threadid_x >= 0 && threadid_y >= 0)
	{
		int pitch = cost.pitch;
		int slicepitch = pitch * w;
		char* point = (char*)cost.ptr;
		int* pos = (int*)(point + threadid_y * slicepitch + threadid_x * pitch);
		int min = *pos;
		int num = 0;
		for (int i = 0; i < pitch / (sizeof(int)); i++)
		{
			if (pos[i] < min)
			{
				min = pos[i];
				num = i;
			}
		}
		img(threadid_y, threadid_x) = make_uchar1(num);
	}	
}

__global__ void Compute_image(cudaPitchedPtr cost, PtrStepSz<uchar1> img)
{
	int threadid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int threadid_y = blockIdx.y*blockDim.y + threadIdx.y;
	int pitch = cost.pitch;
	int slicepitch = pitch * cost.ysize;
	char* point = (char*)cost.ptr;
	int* pos = (int*)(point + threadid_y * slicepitch + threadid_x * pitch);
	int min = *pos;
	int num = 0;
	for (int i = 0; i < pitch / (sizeof(int)); i++)
	{
		if (pos[i] < min)
		{
			min = pos[i];
			num = i;
		}
	}
	img(threadid_y, threadid_x) = make_uchar1(num);
}


/*
void Show_depth::compute_img()
{
	
	img.create(h, w, CV_8UC1);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			int num = 0;
			long min_ = cost[i * d * w + j * d];
			for (int k = 0; k < min<int>(d ,w - k); k++)
			{
				int flag=0;
				if (i == 100 && j == 500 && k == 100)
				{
					flag++;
				}
				if (cost[i * d * w + j * d + k] < min_)
				{
					min_ = cost[i * d * w + j * d + k];
					num = k;
				}
			}
			img.at<uchar>(i, j) = num;
			//delete the bad point
			if (min_ >= 20)
			{
				img.at<uchar>(i, j) = 0;
			}
			else
			{
				img.at<uchar>(i, j) = num;
			}
		}
	}
	//medianBlur(img,img,5);
}
*/
float Show_depth::error_rate()
{
//	cvtColor(depth, depth, CV_RGB2GRAY);
	float num = 0;
	float err = 0;
	float total = 0;
	cout <<"elemsize is:"<< depth.elemSize()<<endl;
	cout << "elemsize1 is:" << depth.elemSize1() << endl;
	float sum = 0;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			total++;
			if (depth.at<unsigned short>(i, j) != 0 && img.at<uchar>(i, j) != 0)
			{
				num++;
				float medium1 = depth.at<unsigned short>(i, j) / 256.0;
				float medium2 = img.at<uchar>(i, j);
				sum += medium2 / medium1;
				if (abs(medium1-medium2) >= 3)
				{
					err++;
				}
			}
		}
	}
	cout << "比例系数为：" << sum / num << endl;
	cout << "错误率为" << err / num << endl;
	cout << "测试数据占全图比例为" << num / total << endl;
	return err / num;
}

void Show_depth::show()
{
	GpuMat gmat(h, w, CV_8UC1);
	img.create(h, w, CV_8UC1);
	dim3 block_size(block_x, block_y);
	dim3 grid_size(grid_x, grid_y);
	gpu_add << <grid_size, block_size >> > (gpu_cost, gmat, w, h);
	cudaError_t status;
//	Compute_image << <block_size, grid_size >> > (gpu_cost, gmat);
	status = cudaThreadSynchronize();
	if (status != cudaSuccess)
	{
		cout << cudaGetErrorString(status);
	}
	gmat.download(img);
	imshow("depth_image",img);
}
