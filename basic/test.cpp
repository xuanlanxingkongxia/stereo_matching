#include<opencv2\opencv.hpp>
#include"IO_KITTI.h"
#include "Census_cost.cuh"
#include "Show_depth.cuh"
#include "Semi_global_matching.cuh"
#include "NCC_cost.h"
#include "MI_cost.h"
#include "Cost_aggregation.cuh"
#include "KITTI_test.h"
#define default_depth 256

int main()
{
	KITTI_test test1 ("D:\\Graduation thesis\\KITTI2015\\training");
	test1.read();
	test1.evaluate();

	IO_KITTI img("D:\\Graduation thesis\\KITTI2015\\training");
	Mat left,right,depth;
	img.read(left, right, depth);
	Mat depth_2;
	depth_2.create(depth.rows, depth.cols, CV_8UC1);
	/*	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
	sgbm->setPreFilterCap(63);
	int sgbmWinSize = 9;//根据实际情况自己设定
	int NumDisparities = 256;//根据实际情况自己设定
	int UniquenessRatio = 10;//根据实际情况自己设定
	sgbm->setBlockSize(sgbmWinSize);
	int cn = left.channels();
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(NumDisparities);
	sgbm->setUniquenessRatio(UniquenessRatio);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(10);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);
	sgbm->compute(left, right, depth_2);
	depth_2.convertTo(depth_2, CV_8U, 255 / (NumDisparities *16.));
	imshow("example",depth_2);*/

	Census_cost the_cost(&left,&right,img.R,img.T);
	the_cost.set_depth(default_depth);
	the_cost.evaluate();
	GpuMat left_g(left.rows,left.cols,CV_8UC3);
	GpuMat right_g(left.rows, left.cols, CV_8UC3);
	left_g.upload(left);
	right_g.upload(right);
//	Cost_aggregation aggr(depth.rows, depth.cols, default_depth, left_g, right_g, the_cost.gpu_cost);
//	aggr.process();
	Semi_global_matching sgm(depth.rows, depth.cols, default_depth, left_g, right_g, the_cost.gpu_cost);
	sgm.process();
	Show_depth show(left.rows, left.cols, default_depth, the_cost.cost, depth,the_cost.gpu_cost);
	cout << "raw cost:";
//	show.compute_img();
	show.show();
	show.error_rate();

/*	Semi_global_matching semi_gl(left.rows, left.cols, default_depth, the_cost.cost);
	semi_gl.set_image(&left, &right);
	semi_gl.process();
	Show_depth show_2(left.rows,left.cols,default_depth,the_cost.cost,depth);
	cout << "semi_global:";
	show_2.compute_img();
	show_2.error_rate();
	show_2.show();*/

/*	for (int i = 0; i < depth.rows; i++)
	{
		for (int j = 0; j < depth.cols; j++)
		{
			depth_2.at<uchar>(i, j) = (depth.at<short>(i, j)) / 256.0;
		}
	}
	imshow("it", depth_2);*/
	waitKey(0);
//	getchar();
//	getchar();
}