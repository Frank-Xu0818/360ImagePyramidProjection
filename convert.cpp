#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

static inline double radians(const double deg) { return deg * CV_PI / 180.0; }

// convertCoordinate from coordinate A to B
// x 是沿所求坐标系转换为原坐标系X轴旋转的角度
// y 是沿所求坐标系转换为原坐标系y轴旋转的角度
// z 是沿所求坐标系转换为原坐标系z轴旋转的角度
// sequence 是矩阵旋转的次序，0是x->y->z，1是x->z->y，2是y->x->z，3是y->z->x，4是z->x->y，5是z->y->x
Mat convertCoordinate(double x, double y, double z, int sequence)
{
	Mat mx = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
	Mat my = (Mat_<double>(3, 3) << cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
	Mat mz = (Mat_<double>(3, 3) << cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);

	Mat A = Mat::zeros(3, 3, CV_64FC1);

	if (sequence == 0)
		A = mz * my * mx;
	else if (sequence == 1)
		A = my * mz * mx;
	else if (sequence == 2)
		A = mz * mx * my;
	else if (sequence == 3)
		A = mx * mz * my;
	else if (sequence == 4)
		A = my * mx * mz;
	else if (sequence == 5)
		A = mx * my * mz;

	return A;
}

Mat A1 = convertCoordinate(radians(0), radians(90), radians(-45), 5);
Mat B1 = convertCoordinate(radians(45), -asin(1.0 / sqrt(3)) - CV_PI / 2.0, radians(0), 2);
Mat A2 = convertCoordinate(radians(0), radians(0), radians(45), 2);
Mat B2 = convertCoordinate(radians(-45), asin(1.0 / sqrt(3)) - CV_PI, radians(0), 2);
Mat A3 = convertCoordinate(radians(180), radians(90), radians(-45), 1);
Mat B3 = convertCoordinate(radians(135), asin(1.0 / sqrt(3)) - CV_PI / 2.0, radians(0), 2);
Mat A4 = convertCoordinate(radians(45), radians(-90), radians(0), 3);
Mat B4 = convertCoordinate(radians(45), asin(1.0 / sqrt(3)) - CV_PI / 2.0, radians(0), 2);

// get x, y, z coords from out image pixels coords
// i, j 是输出图像的xy坐标
// face 是面的号码
// halfOutSize 输出图像宽度的一半
// 将输出图像的xy值转换为三维空间中的xyz坐标，对应成一个坐标值由 -1到1的立方体
//Vec3d outImgToXYZ(int i, int j, int face, int halfOutSize, int toward, 
//	const Mat & A1, const Mat & A2, const Mat & A3, const Mat & A4, 
//	const Mat & B1, const Mat & B2, const Mat & B3, const Mat & B4)
Vec3d outImgToXYZ(float i, float j, int face, int halfOutSize, int toward)
{
	double a, b;
	Vec3d vec(0, 0, 0);

	Mat c = Mat::zeros(3, 1, CV_64FC1);
	Mat d = Mat::zeros(3, 1, CV_64FC1);

	Mat t = Mat::zeros(3, 1, CV_64FC1);

	if (toward == 0) // front face
	{
		a = i * 4.0 / (halfOutSize * 2);
		b = j * 4.0 / (halfOutSize * 2);
		if (face == 0) // down
		{
			a = a - 2.0;
			b = 2.0 - b;

			vec = Vec3d(sqrt(3) - 1, a, b);
		}
		else if (face == 1) // left top
		{
			t.at<double>(0, 0) = b - 1;
			t.at<double>(1, 0) = a - 1;
			t.at<double>(2, 0) = 0;

			//first coordinate conversion
			c = A1 * t;
			c.at<double>(2, 0) = c.at<double>(2, 0) * sqrt(3);  // stretch;

			//second coordinate conversion
			c.at<double>(0, 0) = c.at<double>(0, 0) - (1 - sqrt(3));
			c.at<double>(1, 0) = c.at<double>(1, 0) - 0;
			c.at<double>(2, 0) = c.at<double>(2, 0) - sqrt(2);

			d = B1 * c;

			vec = Vec3d(d.at<double>(0, 0), d.at<double>(1, 0), d.at<double>(2, 0));
		}
		else if (face == 2) // left bottom
		{
			t.at<double>(0, 0) = b - 3;
			t.at<double>(1, 0) = a - 1;
			t.at<double>(2, 0) = 0;

			//first coordinate conversion
			c = A2 * t;
			c.at<double>(0, 0) = c.at<double>(0, 0) * sqrt(3);  // stretch

			//second coordinate conversion
			c.at<double>(0, 0) = c.at<double>(0, 0) - sqrt(2);
			c.at<double>(1, 0) = c.at<double>(1, 0) - 0;
			c.at<double>(2, 0) = c.at<double>(2, 0) - (1 - sqrt(3));

			d = B2 * c;

			vec = Vec3d(d.at<double>(0, 0), d.at<double>(1, 0), d.at<double>(2, 0));
		}
		else if (face == 3) // right top 
		{
			t.at<double>(0, 0) = b - 1;
			t.at<double>(1, 0) = a - 3;
			t.at<double>(2, 0) = 0;

			// first coordinate conversion
			c = A3 * t;
			c.at<double>(2, 0) = c.at<double>(2, 0) * sqrt(3);  // stretch

			// second coordinate conversion
			c.at<double>(0, 0) = c.at<double>(0, 0) - (sqrt(3) - 1);
			c.at<double>(1, 0) = c.at<double>(1, 0) - 0;
			c.at<double>(2, 0) = c.at<double>(2, 0) - sqrt(2);

			d = B3 * c;
			vec = Vec3d(d.at<double>(0, 0), d.at<double>(1, 0), d.at<double>(2, 0));
		}
		else if (face == 4) // right bottom
		{
			t.at<double>(0, 0) = b - 3;
			t.at<double>(1, 0) = a - 3;
			t.at<double>(2, 0) = 0;

			// first coordinate conversion
			c = A4 * t;
			c.at<double>(2, 0) = c.at<double>(2, 0) * sqrt(3);;  // stretch

			// second coordinate conversion
			c.at<double>(0, 0) = c.at<double>(0, 0) - (sqrt(3) - 1);
			c.at<double>(1, 0) = c.at<double>(1, 0) - 0;
			c.at<double>(2, 0) = c.at<double>(2, 0) - sqrt(2);

			d = B4 * c;
			vec = Vec3d(d.at<double>(0, 0), d.at<double>(1, 0), d.at<double>(2, 0));
		}
	}

	return vec;
}

void mapShpereToPyramidCoordinate(
	const float i,
	const float j,
	int face,
	const Size SphereSize,
	int halfOutSize,
	float & srcX,
	float & srcY)
{
	int edge = SphereSize.width / 4;

	Vec3d vec = outImgToXYZ(i, j, face, halfOutSize, 0);
	double theta = atan2(vec[1], vec[0]);	  // 水平方向夹角
	double r = hypot(vec[0], vec[1]);		  // 计算斜边长
	double phi = atan2(vec[2], r);			  // 垂直方向夹角

	// 对应原图像的坐标值
	srcX = (float)(2.0 * edge * (theta + CV_PI) / CV_PI);
	srcY = (float)(2.0 * edge * (CV_PI / 2 - phi) / CV_PI);
}

Mat getConvertMap(const Size SphereSize, const Size PyramidSize)
{
	Mat convertMap = Mat(PyramidSize, CV_32FC2);
	int halfOutSize = PyramidSize.width / 2;

	int face = 0;

	for (int i = 0; i < PyramidSize.width; ++i)
	{
		for (int j = 0; j < PyramidSize.height; ++j)
		{
			face = 0;
			if (fabs(double(halfOutSize - i)) + fabs(double(halfOutSize - j)) <= halfOutSize)
			{
				face = 0;
			}
			else if (i < halfOutSize && j < halfOutSize)
			{
				face = 1;	// 左上角
			}
			else if (i < halfOutSize && j > halfOutSize)
			{
				face = 2;	// 左下角
			}
			else if (i > halfOutSize && j < halfOutSize)
			{
				face = 3;	// 右上角
			}
			else if (i > halfOutSize && j > halfOutSize)
			{
				face = 4;	// 右下角
			}

			float srcX;
			float srcY;
			mapShpereToPyramidCoordinate((float)i, (float)j, face, SphereSize, halfOutSize, srcX, srcY);
			convertMap.at<Point2f>(j, i) = Point2f(srcX, srcY);
		}
	}

	return convertMap;
}

void convert2Pyr(const cv::Mat& SphereImage, Mat & PyramidImage)
{
	Mat convertMap = getConvertMap(SphereImage.size(), PyramidImage.size());

	remap(SphereImage, PyramidImage, convertMap, Mat(), CV_INTER_CUBIC, BORDER_WRAP);
}

// convert using an inverse transformation
void convertBack(const Mat & imgIn, Mat & imgOut)
{
	Size inSize = imgIn.size();
	Size outSize = imgOut.size();

	int edge = inSize.width / 4; // 视角宽度
	int halfOutSize = outSize.width / 2;

	int face = 0;

	for (int i = 0; i < outSize.width; i++)
	{
		for (int j = 0; j < outSize.height; j++)
		{
			face = 0;
			if (fabs(double(halfOutSize - i)) + fabs(double(halfOutSize - j)) <= halfOutSize)
			{
				face = 0;
			}
			else if (i < halfOutSize && j < halfOutSize)
			{
				face = 1;	// 左上角
			}
			else if (i < halfOutSize && j > halfOutSize)
			{
				face = 2;	// 左下角
			}
			else if (i > halfOutSize && j < halfOutSize)
			{
				face = 3;	// 右上角
			}
			else if (i > halfOutSize && j > halfOutSize)
			{
				face = 4;	// 右下角
			}

			Vec3d vec = outImgToXYZ((float)i, (float)j, face, halfOutSize, 0);
			double theta = atan2(vec[1], vec[0]);	  // 水平方向夹角
			double r = hypot(vec[0], vec[1]);		  // 计算斜边长
			double phi = atan2(vec[2], r);			  // 垂直方向夹角

			// 对应原图像的坐标值
			double uf = (2.0 * edge * (theta + CV_PI) / CV_PI);
			double vf = (2.0 * edge * (CV_PI / 2 - phi) / CV_PI);

			int ui = int(uf) % inSize.width;
			int vi = vf < 0 ? 0 : int(vf);
			vi = vi > inSize.height - 1 ? inSize.height - 1 : vi;

			imgOut.at<Vec3b>(j, i) = imgIn.at<Vec3b>(vi, ui);
		}
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("usage : convert.exe  shpere.jpg  pyramid.jpg\n");
		return -1;
	}

	Mat imgIn = imread(argv[1]);

	int outWidth = (int)(imgIn.size().width * sqrt(2) / 4);

	Mat imgOut(Size(outWidth, outWidth), CV_8UC3);

	printf("Converting ...\n"); 
	
	//convertBack(imgIn, imgOut);
	
	convert2Pyr(imgIn, imgOut);

	imwrite(argv[2], imgOut);

	return 0;
}
