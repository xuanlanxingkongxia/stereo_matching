#include "IO_KITTI.h"



IO_KITTI::IO_KITTI(string file_name_)
	:Image_io(file_name_)
{
}


IO_KITTI::~IO_KITTI()
{
}

bool IO_KITTI::read(Mat& left, Mat& right, Mat& depth)
{
	Matx33d R0[6];
	int image_num = 0;
//	srand((unsigned int)time(nullptr));
//	int image_num = rand() % 200;
	string image_name;
	for (int i = 0; i < 6; i++)
	{
		image_name = to_string(image_num % 10) + image_name;
		image_num = image_num / 10;
	}
	string left_name = file_name + "\\image_2\\" + image_name + "_10.png";
	string right_name = file_name + "\\image_3\\" + image_name + "_10.png";
	string depth_name = file_name + "\\disp_noc_0\\" + image_name + "_10.png";
	string calib_name = file_name + "\\calib_cam_to_cam\\" + image_name + ".txt";
	left = imread(left_name);
	right = imread(right_name);
	depth = imread(depth_name, CV_LOAD_IMAGE_UNCHANGED);
	ifstream calib(calib_name);
	char temp[200];
	string try_;
	Vec3d T1(3,1,CV_64FC1);
	Vec3d T2(3,1,CV_64FC1);
	for (int k = 2; k < 4; k++)
	{
		while (string(temp) != string("R_0") + to_string(k) + ":")
		{
			calib >> temp;
		}
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				calib >> R0[(k - 2) * 3](i, j);
			}
		}
		while (string(temp) != "R_rect_0" + to_string(k) + ":")
		{
			calib >> temp;
		}
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				calib >> R0[(k - 2) * 3 + 1](i, j);
			}
		}
		while (string(temp) != "P_rect_0" + to_string(k) + ":")
		{
			calib >> temp;
		}
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				calib >> R0[(k - 2) * 3 + 2](i, j);
			}
			if (k == 2)
			{
				calib >> T1[i];
			}
			else
			{
				calib >> T2[i];
			}
		}
	}
	R = (R0[5] * R0[4] * R0[3]).inv()*(R0[2] * R0[1] * R0[0]);
	T = T2 - R * T1;
	T /= -256.0;
	if (left.empty() || right.empty() || depth.empty())
	{
		cout << "路径有误" << endl;
		return false;
	}
	else
	{
		return true;
	}
}

bool IO_KITTI::read(vector<Mat>& left, vector<Mat>& right, vector<Mat>& depth)
{
	for (int i = 0; i < image_num_sum; i++)
	{
		int image_num = rand() % 200;
		string image_name;
		for (int i = 0; i < 6; i++)
		{
			image_name = to_string(image_num % 10) + image_name;
			image_num = image_num / 10;
		}
		string left_name = file_name + "\\image_2\\" + image_name + "_10.png";
		string right_name = file_name + "\\image_3\\" + image_name + "_10.png";
		string depth_name = file_name + "\\disp_noc_0\\" + image_name + "_10.png";
		left.push_back(imread(left_name));
		right.push_back(imread(right_name));
		depth.push_back(imread(depth_name, CV_LOAD_IMAGE_UNCHANGED));
		if (left[i].empty() || right[i].empty())
		{
			cout << "路径有误" << endl;
			return false;
		}
	}
	return true;
}

void IO_KITTI::save(Mat& d_left, Mat& d_right)
{
	
}

void IO_KITTI::save(vector<Mat>& d_left, vector<Mat>& d_right)
{

}