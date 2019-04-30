#include "image_io.h"



Image_io::Image_io(string file_name_)
{
	file_name = file_name_;
}

Image_io::~Image_io()
{
}

bool Image_io::read(Mat& left, Mat& right ,Mat& depth)
{
	return true;
}

bool Image_io::read(vector<Mat>& left, vector<Mat>& right, vector<Mat>& depth)
{
	return true;
}

void Image_io::set_save_place(string save_name_)
{
	save_name = save_name_;
}

void Image_io::save(Mat& d_left, Mat& d_right)
{
	return;
}

void Image_io::save(vector<Mat>& d_left, vector<Mat>& d_right)
{
	return;
}