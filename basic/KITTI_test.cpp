#include "KITTI_test.h"



KITTI_test::KITTI_test(string filename_)
{
	filename = filename_;
}


KITTI_test::~KITTI_test()
{
}

bool KITTI_test::read()
{
	for (int i = 0; i < 200; i++)
	{
		int image_num = 0;
		//	srand((unsigned int)time(nullptr));
		//	int image_num = rand() % 200;
		string image_name;
		for (int i = 0; i < 6; i++)
		{
			image_name = to_string(image_num % 10) + image_name;
			image_num = image_num / 10;
		}
		string depth_name = filename + "\\disp_noc_0\\" + image_name + "_10.png";
		string new_name = filename + "\\sceneflow_result\\" + image_name + "_10.png";
		Mat new_image = imread(new_name);
		cvtColor(new_image, new_image, CV_RGB2GRAY);
		Mat standard_image = imread(depth_name, CV_LOAD_IMAGE_UNCHANGED);
		image_new.push_back(new_image);
		image_s.push_back(standard_image);
	}
	h = image_new[0].rows;
	w = image_new[0].cols;
	return true;
}

void KITTI_test::evaluate()
{
	double err_sum = 0;
	for (int q = 0; q < 200; q++)
	{
		float num = 0;
		float err = 0;
		float total = 0;
		float sum = 0;
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{
				//total++;
				//cout << image_s[q].at<unsigned short>(i, j);
				//int test000 = image_new[q].at<uchar>(i, j);
				//cout << test000;
				if (image_s[q].at<unsigned short>(i, j) != 0 && image_new[q].at<uchar>(i, j) != 0)
				{
					num++;
					float medium1 = image_s[q].at<unsigned short>(i, j) / 256.0;
					float medium2 = image_new[q].at<uchar>(i, j);
					sum += medium2 / medium1;
					if (abs(medium1 - medium2) >= 3)
					{
						err++;
					}
				}
			}
		}
		err_sum += err / num;
	}
	cout << "´íÎóÂÊÎª:" << err_sum / 200 << endl;
}