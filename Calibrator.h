#ifndef CALIBRATOR_H_
#define CALIBRATOR_H_

#include <fstream>
#include <opencv2\opencv.hpp>

using namespace cv;

class Calibrator {
public:
	Calibrator();
	Calibrator(const char* filename);
	Mat Undistort(const Mat& fisheye);
	bool Valid(int x, int y);
private:
	char* name;
	Mat map1, map2, mask;
};

#endif

