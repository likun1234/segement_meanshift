#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

static void help(char** argv)
{
	cout << "\nDemonstrate mean-shift based color segmentation in spatial pyramid.\n"
		<< "Call:\n   " << argv[0] << " image\n"
		<< "This program allows you to set the spatial and color radius\n"
		<< "of the mean shift window as well as the number of pyramid reduction levels explored\n"
		<< endl;
}

static void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) )
{
	CV_Assert( !img.empty() );
	RNG rng = theRNG();
	Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
	for( int y = 0; y < img.rows; y++ )
	{
		for( int x = 0; x < img.cols; x++ )
		{
			if( mask.at<uchar>(y+1, x+1) == 0 )
			{
				Scalar newVal( rng(256), rng(256), rng(256) );
				floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );
			}
		}
	}
}

void meanshift(Mat &ori)
{
	clock_t start, finish;
	start = clock();
	Mat res = Mat::zeros(ori.size(),CV_8UC3);
	int spatialRad = 15;    
	int colorRad = 15;   
	int maxPyrLevel = 1;   
	if (ori.empty()){
		vector<cv::Point2f> pp;
	}
	imshow("original img",ori);
	waitKey(0);
	pyrMeanShiftFiltering( ori, res, spatialRad, colorRad, maxPyrLevel );
	imshow( "mean_shift_res", res );
	waitKey(0);
	floodFillPostprocess( res, Scalar::all(2) );
	imwrite( "/home/lk/project/segment_new/mean_shift.jpg", res );
	imshow( "floodfill", res );
	waitKey(0);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
}


Mat convertTo3Channels(const Mat& binImg)
{
	Mat three_channel = Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
	vector<Mat> channels;
	for (int i=0;i<3;i++)
	{
		channels.push_back(binImg);
	}
	merge(channels,three_channel);
	return three_channel;
}


int lapulas_img(Mat &img)
{
	if (img.empty())
	{
		std::cout<<"no img input!"<<std::endl;
		return -1;
	}
	imshow("ori",img);
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	cv::filter2D(img,img,CV_8UC3,kernel);
	imshow("pro",img);
	cv::waitKey(0);
	return 0;
}

int log_ac(Mat &image)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			image.at<Vec3f>(i, j)[0] = log(1 + image.at<Vec3b>(i, j)[0]);
			image.at<Vec3f>(i, j)[1] = log(1 + image.at<Vec3b>(i, j)[1]);
			image.at<Vec3f>(i, j)[2] = log(1 + image.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255
	normalize(image, image, 0, 255, CV_MINMAX);
	//转换成8bit图像显示
	cv::convertScaleAbs(image, image);
	imshow("Soure", image);
	imshow("after", image);
	//waitKey();
	//return 0;
}

Mat gamma_acc(Mat &image)
{
	Mat imageGamma(image.size(), CV_32FC3);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			imageGamma.at<Vec3f>(i, j)[0] = (image.at<Vec3b>(i, j)[0])*(image.at<Vec3b>(i, j)[0])*(image.at<Vec3b>(i, j)[0]);
			imageGamma.at<Vec3f>(i, j)[1] = (image.at<Vec3b>(i, j)[1])*(image.at<Vec3b>(i, j)[1])*(image.at<Vec3b>(i, j)[1]);
			imageGamma.at<Vec3f>(i, j)[2] = (image.at<Vec3b>(i, j)[2])*(image.at<Vec3b>(i, j)[2])*(image.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255
	normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);
	//转换成8bit图像显示
	convertScaleAbs(imageGamma, imageGamma);
	imshow("伽马变换图像增强效果", imageGamma);
	waitKey();
	return imageGamma;
}
void pic_pro(Mat &img, Mat &img_canny_close)
{
	Mat gray,img_canny;
	//cv::resize(img,img,Size(540,720));
	Mat img1 = gamma_acc(img);//gamma_acc
	cvtColor(img1,gray,cv::COLOR_BGR2GRAY);
	cv::Canny(gray,img_canny,30,250);
	Mat kernel = getStructuringElement(cv::MORPH_RECT,Size(5,5));
	cv::morphologyEx(img_canny,img_canny_close,cv::MORPH_CLOSE,kernel);
	cv::bitwise_not(img_canny_close,img_canny_close);
	cv::imwrite("/home/lk/project/segment_new/result.png",img_canny_close);

}
int main(int argc, char *argv[])
{
	Mat img = cv::imread("/home/lk/project/new_segment/22.png");
	Mat pro_result;
	pic_pro(img, pro_result);
	imshow("img_ori",img);
	cv::waitKey(0);
	Mat pic_pro = convertTo3Channels(pro_result);
	meanshift(pic_pro);
	return 0;

}
