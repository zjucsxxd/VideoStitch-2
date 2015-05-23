#include "Calibrator.h"

Calibrator::Calibrator() {
}

Calibrator::Calibrator(const char* filename) {
	FILE* file = fopen(filename, "rb");
	int h, w;
	fread(&w, sizeof(int), 1, file);
	fread(&h, sizeof(int), 1, file);
	map1 = Mat(h, w, CV_32FC1);
	map2 = Mat(h, w, CV_32FC1);
	fread(map1.data, sizeof(float), h * w, file);
	fread(map2.data, sizeof(float), h * w, file);
	fclose(file);
}

Mat Calibrator::Undistort(const Mat& fisheye) {
	Mat m(map1.rows, map1.cols, CV_8UC3);
	mask = Mat(map1.rows, map1.cols, CV_8UC1);
	memset(mask.data, 0, sizeof(char)* map1.rows * map1.cols);
	remap(fisheye, m, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
	for (int i = 0; i < map1.rows; ++i) {
		for (int j = 0; j < map2.cols; ++j) {
			float t1 = ((float*)map1.data)[i * map2.cols + j];
			float t2 = ((float*)map2.data)[i * map2.cols + j];
			if (((float*)map1.data)[i * map2.cols + j] < 0 || ((float*)map1.data)[i * map2.cols + j] >= 1920
				|| ((float*)map2.data)[i * map2.cols + j] < 0 || ((float*)map2.data)[i * map2.cols + j] >= 1080) {
				mask.data[(i * map2.cols + j)] = 1;
			}
		}
	}
	return m;
}

bool Calibrator::Valid(int x, int y) {
	if (y < 0 || y >= map2.rows || x < 0 || x >= map2.cols)
		return false;
	return (mask.data[y * map2.cols + x] == 0);
}