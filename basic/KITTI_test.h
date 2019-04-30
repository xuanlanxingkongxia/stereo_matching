#pragma once
#include<string>
#include<vector>
#include<opencv2\opencv.hpp>
#include<fstream>
#include<iostream>
using namespace std;
using namespace cv;

class KITTI_test
{
public:
	KITTI_test(string filename_);
	~KITTI_test();

	bool read();
	void evaluate();

private:
	string filename;
	vector<Mat> image_s;
	vector<Mat> image_new;
	int h;
	int w;
};

