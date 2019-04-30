#pragma once
#include "image_io.h"
#include<time.h>

class IO_KITTI :
	public Image_io
{
public:
	IO_KITTI(string file_name_);
	~IO_KITTI();
	
	bool read(Mat& left, Mat& right, Mat& depth) override;
	bool read(vector<Mat>& left, vector<Mat>& right, vector<Mat>& depth) override;
	void save(Mat& d_left, Mat& d_right) override;
	void save(vector<Mat>& d_left, vector<Mat>& d_right) override;
	Matx33d R;
	Vec3d T;

protected:
	int image_num_sum = 200;
};

