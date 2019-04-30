#pragma once
#include<string>
#include<vector>
#include<opencv2\opencv.hpp>
#include<fstream>
#include<iostream>
using namespace std;
using namespace cv;
class Image_io
{
public:
	Image_io(string file_name_);
	~Image_io();

	//¶ÁÐ´º¯Êý
	virtual bool read(Mat& left ,Mat& right,Mat& depth);
	virtual bool read(vector<Mat>& left, vector<Mat>& right, vector<Mat>& depth);
	void set_save_place(string save_name_);
	virtual void save(Mat& d_left, Mat& d_right);
	virtual void save(vector<Mat>& d_left ,vector<Mat>& d_right);

protected:
	string file_name;
	string save_name;
};

