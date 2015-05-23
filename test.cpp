#include "test.h"
using namespace cv;
void matSet(CvMat *mat, double* value) {
	int k = 0;
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			cvmSet(mat, i, j, value[k++]);
		}
	}
}

void printMatrix(CvMat *mat) {
	int k = 0;
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			printf("%10.3f", cvmGet(mat, i, j));
		}
		printf("\n");
	}
}

void testit() {
	VideoCapture vc1("dataset\\view1\\2.MP4");
	VideoCapture vc2("dataset\\view2\\2.MP4");
	VideoCapture vc3("dataset\\view3\\2.MP4");
	VideoCapture vc4("dataset\\view4\\2.MP4");
	Mat frame1, frame2, frame3, frame4;
	vc1 >> frame1;
	vc2 >> frame2;
	vc3 >> frame3;
	vc4 >> frame4;
	imwrite("frame1.png", frame1);
	imwrite("frame2.png", frame2);
	imwrite("frame3.png", frame3);
	imwrite("frame4.png", frame4);
}