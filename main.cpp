#include <opencv2\opencv.hpp>
#include <opencv2\stitching\stitcher.hpp>
#include <iostream>
#include <vector>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include "Calibrator.h"
#include <iostream>
#include "test.h"
using namespace std;
using namespace cv;

#define PI 3.1415926536

#define PX_EVEN 250
#define PY_EVEN 810
#define PY_ODD 618 
#define PY_WIDTH 28

int g_width, g_height, g_f;
double horizontal_theta = 117 / 180.0 * PI * 0.5;

Calibrator calibrator("distort.dat");

void MapCoordToAngle(double x, double y, double f, double& phi, double& theta, int i) {
	if (i % 2 == 0) {
		x -= g_width / 2;
		y -= g_height / 2;
		theta = atan(x / f);
		phi = atan(y / sqrt(f * f + x * x));
	}
	else {
		double u0 = (x + 0.0) / g_width;
		double v0 = (y + 0.0) / g_height;
		if (i == 1) {
			v0 = 1 - v0;
		}
		else {
			u0 = 1 - u0;
		}
		double u = v0 * g_height;
		double v = u0 * g_width;
		theta = atan((u - g_height * 0.5) / f);
		phi = atan((v - g_width * 0.5) / f * cos(theta));
	}
	theta += i * 0.5 * PI;
	if (theta < 0 && i == 0) {
		theta += 2 * PI;
	}
}

void MapAngleToCoord(double phi, double theta, double f, int& x, int &y, int i) {
	theta -= i * 0.5 * PI;
	if (theta > PI && i == 0)
		theta -= 2 * PI;
	int h = g_height, w = g_width;
	if (i % 2 == 1) {
		int k = h;
		h = w, w = k;
	}
	if (theta < -PI / 2 || theta > PI / 2) {
		x = -1; y = -1;
		return;
	}
	double t = tan(theta);
	double u = f * tan(theta) + w * 0.5;
	double v = f / cos(theta) * tan(phi) + h * 0.5;
	if (i % 2 == 0) {
		x = (u + 0.5);
		y = (v + 0.5);
	}
	else
	if (i == 3) {
		double u0 = 1 - v / h;
		double v0 = u / w;
		x = u0 * g_width + 0.5;
		y = v0 * g_height + 0.5;
	}
	else {
		double u0 = v / h;
		double v0 = 1 - u / w;
		x = u0 * g_width + 0.5;
		y = v0 * g_height + 0.5;
	}
	if (x < 0 || x >= g_width || y < 0 || y >= g_height) {
		x = -1, y = -1;
	}
}

bool SamplePixel(Mat frame, double phi, double theta, uchar* color, int i) {
	int x = 0, y = 0;
	MapAngleToCoord(phi, theta, g_f, x, y, i);
	if (x > 0) {
		if (!calibrator.Valid(x, y))
			return false;
		memcpy(color, frame.data + (y * frame.cols + x) * 3, sizeof(uchar)* 3);
		return true;
	}
	return false;
}

void Stitch(Mat pano, vector<Mat> frames, double* offset_phi, double* offset_theta, double s_x) {
	for (int i = 0; i < pano.rows; ++i) {
		for (int j = 0; j < pano.cols; ++j) {
			double t1 = i / (pano.rows + 0.0), t2 = j / (pano.cols + 0.0);
			double gphi = ((i / (pano.rows + 0.0)) - 0.5) * PI;
			double gtheta = (j / (pano.cols + 0.0)) * 2 * PI;
			int ind[] = { 0, 2, 1, 3 };
			for (int index = 0; index < 4; ++index) {
				if (ind[index] != 0 || gtheta < PI)
					SamplePixel(frames[ind[index]], gphi - offset_phi[ind[index]], gtheta * s_x - offset_theta[ind[index]], pano.data + (i * pano.cols + j) * 3, ind[index]);
				else
					SamplePixel(frames[ind[index]], gphi - offset_phi[ind[index]], 2 * PI + (gtheta - 2 * PI) * s_x - offset_theta[ind[index]], pano.data + (i * pano.cols + j) * 3, ind[index]);
			}
//			for (int k = 0; k < 3; ++k) {
//				pano.data[(i * pano.cols + j) * 3 + k] = num * 80;
//			}
/*			int index = 0;
			if (gtheta >= PI * 1.7) {
				gtheta -= PI * 2;
			} else
			if (gtheta > PI * 1.3 && gtheta < PI * 1.7) {
				index = 3;
				gtheta -= PI * 1.5;
			}
			else if (gtheta > PI * 0.7) {
				index = 2;
				gtheta -= PI;
			}
			else if (gtheta > PI * 0.3) {
				index = 1;
				gtheta -= PI * 0.5;
			}
			if (SamplePixel(frames[index], gphi, gtheta, pano.data + (i * pano.cols + j) * 3, index))
				num = 1;
			for (int k = 0; k < 3; ++k) {
				pano.data[(i * pano.cols + j) * 3 + k] = num * 80;
			}*/
		}
	}
}

void FindHomography(vector<Point2f>& obj, vector<Point2f>& scene, Mat& res, int iteration = 2000, double threshold = 2500, int num = 4) {
	num = obj.size();
	iteration = 0;
	while (num > 4) {
		iteration++;
		num = obj.size();
		Mat T(2 * num, 8, CV_64FC1), tinv(8, 2 * num, CV_64FC1), transform(3, 3, CV_64F);
		((double*)transform.data)[8] = 1;
		int max_match = -1;
		memset(T.data, 0, sizeof(double)* 16 * num);
		for (int i = 0; i < num; ++i) {
			((double*)T.data)[i * 8 + 2] = 1;
			((double*)T.data)[(i + num) * 8 + 5] = 1;
		}
		Mat B(2 * num, 1, CV_64FC1), h(8, 1, CV_64FC1);
		for (int k = 0; k < num; ++k) {
			((double*)T.data)[k * 8] = obj[k].x;
			((double*)T.data)[k * 8 + 1] = obj[k].y;
			((double*)T.data)[k * 8 + 6] = -obj[k].x * scene[k].x;
			((double*)T.data)[k * 8 + 7] = -obj[k].y * scene[k].x;
			((double*)T.data)[(k + num) * 8 + 3] = obj[k].x;
			((double*)T.data)[(k + num) * 8 + 4] = obj[k].y;
			((double*)T.data)[(k + num) * 8 + 6] = -obj[k].x * scene[k].y;
			((double*)T.data)[(k + num) * 8 + 7] = -obj[k].y * scene[k].y;
			((double*)B.data)[k] = scene[k].x;
			((double*)B.data)[k + num] = scene[k].y;
		}
		solve(T, B, h, CV_SVD);
		memcpy(transform.data, h.data, sizeof(double)* 8);
		double max_dis = 1e30;
		vector<double> dis(obj.size());
		vector<Point2f> obj1, scene1;
		for (int k = 0; k < obj.size(); ++k) {
			double x[3] = { 0 };
			for (int l = 0; l < 3; ++l) {
				x[l] += ((double*)transform.data)[l * 3] * obj[k].x + ((double*)transform.data)[l * 3 + 1] * obj[k].y + ((double*)transform.data)[l * 3 + 2];
			}
			x[0] /= x[2];
			x[1] /= x[2];
			dis[k] = sqrt((x[0] - scene[k].x) * (x[0] - scene[k].x) + (x[1] - scene[k].y) * (x[1] - scene[k].y));
			if (dis[k] < max_dis)
				max_dis = dis[k];
		}
		for (int k = 0; k < obj.size(); ++k) {
			if (dis[k] < 3 * max_dis) {
				obj1.push_back(obj[k]);
				scene1.push_back(scene[k]);
			}
		}
		if (obj1.size() == 0 || max_dis < 10) {
			obj = obj1;
			scene = scene1;
			memcpy(res.data, transform.data, sizeof(double)* 9);
			break;
		}
	}
	return;
}

void PerspectiveTransform(vector<Point2f>& obj, vector<Point2f>& scene, Mat H) {
	scene.resize(obj.size());
	for (int i = 0; i < obj.size(); ++i) {
		scene[i].x = ((double*)H.data)[0] * obj[i].x + ((double*)H.data)[1] * obj[i].y + ((double*)H.data)[2];
		scene[i].y = ((double*)H.data)[3] * obj[i].x + ((double*)H.data)[4] * obj[i].y + ((double*)H.data)[5];
		double z = ((double*)H.data)[6] * obj[i].x + ((double*)H.data)[7] * obj[i].y + 1;
		scene[i].x /= z;
		scene[i].y /= z;
	}
}

void Rectify21(Mat& res2, Mat& res3, double& offset_phi, double& offset_theta) {
	int offsetX = -res2.cols / 2 + PY_EVEN;
	int offsetY = -res3.cols + PY_ODD;
	int offsetZ = -res2.rows + PX_EVEN;
	Mat res2_part = res2(Rect(-offsetX, -offsetZ, PY_EVEN * 2, PX_EVEN));
	Mat res3_part = res3(Rect(-offsetY, 0, PY_ODD, res3.rows));
	imwrite("res2_21.png", res2_part);
	imwrite("res3_21.png", res3_part);
	int minHessian = 400;

	SurfFeatureDetector detector;

	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	keypoints_object.reserve(10000);
	keypoints_scene.reserve(10000);

	detector.detect(res2_part, keypoints_object);
	detector.detect(res3_part, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(res2_part, keypoints_object, descriptors_object);
	extractor.compute(res3_part, keypoints_scene, descriptors_scene);
	int t = descriptors_object.depth();
/*	Mat output1, output2;
	drawKeypoints(res2_part, keypoints_object, output1);
	drawKeypoints(res3_part, keypoints_scene, output2);
	imwrite("output1.png", output1);
	imwrite("output2.png", output2);

	for (int i = 0; i < keypoints_object.size(); ++i) {
		double phi, theta;
		int x = keypoints_object[i].pt.x - offsetX;
		int y = keypoints_object[i].pt.y - offsetZ;
		MapCoordToAngle(x, y, g_f, phi, theta, 1);
		MapAngleToCoord(phi, theta, g_f, x, y, 0);
		if (x >= 0) {
			keypoints_object[i].pt.x = x + offsetY;
			keypoints_object[i].pt.y = y;
		}
		else {
			keypoints_object[i].pt.x = 0;
			keypoints_object[i].pt.y = 0;
		}
	}
	drawKeypoints(res3_part, keypoints_object, output2);
	imwrite("output3.png", output2);*/
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	BFMatcher matcher;
	std::vector< DMatch > matches;
	matches.reserve(10000);
	matcher.match(descriptors_object, descriptors_scene, matches);

	double min_dist = 100, max_dist = 0;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches, gm;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			int x = keypoints_object[matches[i].queryIdx].pt.x - offsetX;
			int y = keypoints_object[matches[i].queryIdx].pt.y - offsetZ + PY_WIDTH;
			if (calibrator.Valid(x, y))
				good_matches.push_back(matches[i]);
		}
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt + Point2f(-offsetX, -offsetZ));
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt + Point2f(-offsetY, 0));
	}

	Mat img_matches;
	drawMatches(res2_part, keypoints_object, res3_part, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("matches21.png", img_matches);

	vector<double> dphi(obj.size()), dtheta(obj.size()), hashmap(obj.size(), 1);
	for (int i = 0; i < obj.size(); ++i) {
		double tphi4, ttheta4, tphi3, ttheta3;
		MapCoordToAngle(obj[i].x, obj[i].y, g_f, tphi4, ttheta4, 1);
		MapCoordToAngle(scene[i].x, scene[i].y, g_f, tphi3, ttheta3, 0);
		dphi[i] = tphi3 - tphi4;
		dtheta[i] = ttheta3 - ttheta4;
		if (abs(dphi[i]) > 0.17 || abs(dtheta[i]) > 0.25)
			hashmap[i] = 0;
		else
			gm.push_back(good_matches[i]);
	}
	drawMatches(res2_part, keypoints_object, res3_part, keypoints_scene,
		gm, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("matches21_1.png", img_matches);

	while (true) {
		double aphi = 0, atheta = 0;
		int count = 0;
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				count++;
				aphi += dphi[j];
				atheta += dtheta[j];
			}
		}
		if (count == 0) {
			break;
		}
		aphi /= count;
		atheta /= count;
		offset_phi = aphi;
		offset_theta = atheta;
		if (count == 1)
			break;
		double max_dis = 1e-30;
		vector<double> dis(obj.size());
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				dis[j] = sqrt((aphi - dphi[j])*(aphi - dphi[j]) + (atheta - dtheta[j])*(atheta - dtheta[j]));
				if (dis[j] > max_dis)
					max_dis = dis[j];
			}
		}
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j] && dis[j] > max_dis * 0.9) {
				hashmap[j] = 0;
			}
		}
	}
}

void Rectify41(Mat& res2, Mat& res3, double& offset_phi, double& offset_theta) {
	int offsetX = -res2.cols / 2 + PY_EVEN;
	int offsetZ = -res2.rows + PX_EVEN;
	Mat res2_part = res2(Rect(-offsetX, -offsetZ, PY_EVEN * 2, PX_EVEN));
	Mat res3_part = res3(Rect(0, 0, PY_ODD, res3.rows));
	imwrite("res2_41.png", res2_part);
	imwrite("res3_41.png", res3_part);

	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	keypoints_object.reserve(10000);
	keypoints_scene.reserve(10000);

	detector.detect(res2_part, keypoints_object);
	detector.detect(res3_part, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(res2_part, keypoints_object, descriptors_object);
	extractor.compute(res3_part, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matches.reserve(10000);
	matcher.match(descriptors_object, descriptors_scene, matches);

	double min_dist = 100, max_dist = 0;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			int x = keypoints_object[matches[i].queryIdx].pt.x - offsetX;
			int y = keypoints_object[matches[i].queryIdx].pt.y - offsetZ;
			if (calibrator.Valid(x, y))
				good_matches.push_back(matches[i]);
		}
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt + Point2f(-offsetX, -offsetZ));
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat img_matches;
	drawMatches(res2_part, keypoints_object, res3_part, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("matches41.png", img_matches);

	vector<double> dphi(obj.size()), dtheta(obj.size()), hashmap(obj.size(), 1);
	for (int i = 0; i < obj.size(); ++i) {
		double tphi4, ttheta4, tphi3, ttheta3;
		MapCoordToAngle(obj[i].x, obj[i].y, g_f, tphi4, ttheta4, 3);
		MapCoordToAngle(scene[i].x, scene[i].y, g_f, tphi3, ttheta3, 0);
		dphi[i] = tphi3 - tphi4;
		dtheta[i] = ttheta3 - ttheta4;
		if (abs(dphi[i]) > 0.03 || abs(dtheta[i]) > 0.005)
			hashmap[i] = 0;
	}
	while (true) {
		double aphi = 0, atheta = 0;
		int count = 0;
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				count++;
				aphi += dphi[j];
				atheta += dtheta[j];
			}
		}
		if (count == 0) {
			break;
		}
		aphi /= count;
		atheta /= count;
		offset_phi = aphi;
		offset_theta = atheta;
		if (count == 1)
			break;
		double max_dis = 1e-30;
		vector<double> dis(obj.size());
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				dis[j] = sqrt((aphi - dphi[j])*(aphi - dphi[j]) + (atheta - dtheta[j])*(atheta - dtheta[j]));
				if (dis[j] > max_dis)
					max_dis = dis[j];
			}
		}
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j] && dis[j] > max_dis * 0.9) {
				hashmap[j] = 0;
			}
		}
	}
}


void Rectify23(Mat& res2, Mat& res3, double& offset_phi, double& offset_theta) {
	int offsetX = -res2.cols / 2 + PY_EVEN;
	Mat res2_part = res2(Rect(-offsetX, 0, PY_EVEN * 2, PX_EVEN));
	Mat res3_part = res3(Rect(0, 0, PY_ODD, res3.rows));
	imwrite("res2_23.png", res2_part);
	imwrite("res3_23.png", res3_part);

	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	keypoints_object.reserve(10000);
	keypoints_scene.reserve(10000);

	detector.detect(res2_part, keypoints_object);
	detector.detect(res3_part, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(res2_part, keypoints_object, descriptors_object);
	extractor.compute(res3_part, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matches.reserve(10000);
	matcher.match(descriptors_object, descriptors_scene, matches);
	
	double min_dist = 100, max_dist = 0;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches, gm;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			int x = keypoints_object[matches[i].queryIdx].pt.x - offsetX;
			int y = keypoints_object[matches[i].queryIdx].pt.y - PY_WIDTH;
			if (calibrator.Valid(x, y))
				good_matches.push_back(matches[i]);
		}
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt + Point2f(-offsetX, 0));
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat img_matches;
	drawMatches(res2_part, keypoints_object, res3_part, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("matches23.png", img_matches);

	vector<double> dphi(obj.size()), dtheta(obj.size()), hashmap(obj.size(), 1);
	for (int i = 0; i < obj.size(); ++i) {
		double tphi4, ttheta4, tphi3, ttheta3;
		MapCoordToAngle(obj[i].x, obj[i].y, g_f, tphi4, ttheta4, 1);
		MapCoordToAngle(scene[i].x, scene[i].y, g_f, tphi3, ttheta3, 2);
		dphi[i] = tphi3 - tphi4;
		dtheta[i] = ttheta3 - ttheta4;
		if (abs(dphi[i]) > 0.005 || abs(dtheta[i]) > 0.03)
			hashmap[i] = 0;
		else
			gm.push_back(good_matches[i]);
	}
	drawMatches(res2_part, keypoints_object, res3_part, keypoints_scene,
		gm, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("matches23_1.png", img_matches);

	while (true) {
		double aphi = 0, atheta = 0;
		int count = 0;
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				count++;
				aphi += dphi[j];
				atheta += dtheta[j];
			}
		}
		if (count == 0) {
			break;
		}
		aphi /= count;
		atheta /= count;
		offset_phi = aphi;
		offset_theta = atheta;
		if (count == 1)
			break;
		double max_dis = 1e-30;
		vector<double> dis(obj.size());
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				dis[j] = sqrt((aphi - dphi[j])*(aphi - dphi[j]) + (atheta - dtheta[j])*(atheta - dtheta[j]));
				if (dis[j] > max_dis)
					max_dis = dis[j];
			}
		}
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j] && dis[j] > max_dis * 0.9) {
				hashmap[j] = 0;
			}
		}
	}
}

void Rectify43(Mat& res4, Mat& res3, double& offset_phi, double& offset_theta) {
	int offsetX = -res4.cols / 2 + PY_EVEN;
	int offsetY = -res3.cols + PY_ODD;
	Mat res2_part = res4(Rect(-offsetX, 0, PY_EVEN * 2, PX_EVEN));
	Mat res3_part = res3(Rect(-offsetY, 0, PY_ODD, res3.rows));
	imwrite("res2_43.png", res2_part);
	imwrite("res3_43.png", res3_part);

	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	keypoints_object.reserve(10000);
	keypoints_scene.reserve(10000);
	detector.detect(res2_part, keypoints_object);
	detector.detect(res3_part, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(res2_part, keypoints_object, descriptors_object);
	extractor.compute(res3_part, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matches.reserve(10000);
	matcher.match(descriptors_object, descriptors_scene, matches);
	double min_dist = 100, max_dist = 0;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			int x = keypoints_object[matches[i].queryIdx].pt.x - offsetX;
			int y = keypoints_object[matches[i].queryIdx].pt.y;
			if (calibrator.Valid(x, y))
				good_matches.push_back(matches[i]);
		}
	}

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	vector<double> dphi, dtheta, hashmap;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt + Point2f(-offsetX, 0));
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt + Point2f(-offsetY, 0));
	}
	Mat img_matches;
	drawMatches(res2_part, keypoints_object, res3_part, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("matches43.png", img_matches);

	dphi.resize(obj.size());
	dtheta.resize(obj.size());
	hashmap.resize(obj.size(), 1);
	for (int i = 0; i < obj.size(); ++i) {
		double tphi4, ttheta4, tphi3, ttheta3;
		MapCoordToAngle(obj[i].x, obj[i].y, g_f, tphi4, ttheta4, 3);
		MapCoordToAngle(scene[i].x, scene[i].y, g_f, tphi3, ttheta3, 2);
		dphi[i] = tphi3 - tphi4;
		dtheta[i] = ttheta3 - ttheta4;
		if (abs(dphi[i]) > 0.03 || abs(dtheta[i]) > 0.005)
			hashmap[i] = 0;
	}
	while (true) {
		double aphi = 0, atheta = 0;
		int count = 0;
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				count++;
				aphi += dphi[j];
				atheta += dtheta[j];
			}
		}
		if (count == 0)
			break;
		aphi /= count;
		atheta /= count;
		offset_phi = aphi;
		offset_theta = atheta;
		if (count == 1)
			break;
		double max_dis = 1e-30;
		vector<double> dis(obj.size());
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j]) {
				dis[j] = sqrt((aphi - dphi[j])*(aphi - dphi[j]) + (atheta - dtheta[j])*(atheta - dtheta[j]));
				if (dis[j] > max_dis)
					max_dis = dis[j];
			}
		}
		for (int j = 0; j < obj.size(); ++j) {
			if (hashmap[j] && dis[j] >= max_dis * 0.9) {
				hashmap[j] = 0;
			}
		}
	}
	return;
}

int main() {
	testit();
	Mat im1, im2, im3, im4;
	im1 = imread("frame3.png");
	im2 = imread("frame4.png");
	im3 = imread("frame1.png");
	im4 = imread("frame2.png");
	Mat res1, res2, res3, res4;
	res1 = calibrator.Undistort(im1);
	res2 = calibrator.Undistort(im2);
	res3 = calibrator.Undistort(im3);
	res4 = calibrator.Undistort(im4);
	imwrite("res1.png", res1);
	imwrite("res2.png", res2);
	imwrite("res3.png", res3);
	imwrite("res4.png", res4);
	g_width = res1.cols;
	g_height = res1.rows;
	g_f = g_width * 0.5 / tan(horizontal_theta);

	double offset_phi[4] = { 0 }, offset_theta[4] = { 0 };
	Rectify21(res2, res1, offset_phi[1], offset_theta[1]);
	Rectify41(res4, res1, offset_phi[3], offset_theta[3]);
	Rectify23(res2, res3, offset_phi[2], offset_theta[2]);
	Rectify43(res4, res3, offset_phi[0], offset_theta[0]);
	offset_phi[2] = offset_phi[1] - offset_phi[2];
	offset_theta[2] = offset_theta[1] - offset_theta[2];
	offset_theta[0] += offset_theta[2];
	offset_phi[0] += offset_phi[2];
	ofstream os("rectify.txt");
	double s_x = (offset_theta[0] - offset_theta[3] + 2 * PI) / (2 * PI);
	os << s_x << " ";
	offset_theta[3] = offset_theta[0];
	offset_phi[3] = offset_phi[0];
	offset_theta[0] = 0;
	offset_phi[0] = 0;
	for (int i = 0; i < 4; ++i)
		os << offset_theta[i] << " " << offset_phi[i] << " ";
/*	Mat H;
	findHomography(obj, scene, H);

	obj.resize(4);
	obj[0].x = 0; obj[0].y = 0;
	obj[1].x = res2.cols; obj[1].y = 0;
	obj[2].x = res2.cols; obj[2].y = res2.rows;
	obj[3].x = 0; obj[3].y = res2.rows;

	perspectiveTransform(obj, scene, H);
	int x_offset = 0, y_offset = 0, w = res3.cols - 1, h = res3.rows - 1;
	for (int i = 0; i < scene.size(); ++i) {
		if (scene[i].x < x_offset) {
			x_offset = scene[i].x;
		}
		if (scene[i].x > w) {
			w = scene[i].x;
		}
		if (scene[i].y < y_offset) {
			y_offset = scene[i].y;
		}
		if (scene[i].y > h) {
			h = scene[i].y;
		}
	}

	Mat output(h + 1 - y_offset, w + 1 - x_offset, CV_8UC3);
	for (int i = 0; i < res3.rows; ++i) {
		for (int j = 0; j < res3.cols; ++j) {
			int index = (i * res3.cols) + j;
			memcpy(output.data + ((i - y_offset) * output.cols + (j - x_offset)) * 3, res3.data + index * 3, sizeof(uchar)* 3);
		}
	}
	coord.resize((h - y_offset + 1) * (w - x_offset + 1));
	scene.clear();
	for (int i = y_offset; i <= h; ++i) {
		for (int j = x_offset; j <= w; ++j) {
			coord[(i - y_offset) * (w - x_offset + 1) + j - x_offset].x = j;
			coord[(i - y_offset) * (w - x_offset + 1) + j - x_offset].y = i;
		}
	}
	perspectiveTransform(coord, scene, H.inv());
	*/
	vector<Mat> frames;
	frames.push_back(res1);
	frames.push_back(res2);
	frames.push_back(res3);
	frames.push_back(res4);
	Mat pano(1000, 4000, CV_8UC3);
	Stitch(pano, frames, offset_phi, offset_theta, s_x);
	imwrite("pano.png", pano);


/*	for (int i = y_offset; i <= h; ++i) {
		for (int j = x_offset; j <= w; ++j) {
			int index = (i - y_offset) * (w - x_offset + 1) + j - x_offset;
			int ii = scene[index].y;
			int jj = scene[index].x;
			if (ii < 0 || ii >= res2.rows || jj < 0 || jj >= res2.cols)
				continue;
			memcpy(output.data + ((i - y_offset) * output.cols + (j - x_offset)) * 3, res2.data + ((ii * res3.cols) + jj) * 3, sizeof(uchar)* 3);
		}
	}
*/
//	imwrite("output.png", output);
	return 0;
}
