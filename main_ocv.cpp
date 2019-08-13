#include <opencv2/opencv.hpp>

const char* image_files[] = {
    "./gray/1.jpg", "./gray/2.jpg", "./gray/3.jpg",
    "./gray/4.jpg", "./gray/5.jpg", "./gray/6.jpg",
    "./gray/7.jpg", "./gray/8.jpg", "./gray/0.jpg",
    NULL
};

int main()
{
	for (int i=0; i<9; i++) {
		cv::Mat img = cv::imread(image_files[i]);
		cv::imshow("original", img);

	    cv::Mat grad_x, grad_y;
	    cv::Mat abs_grad_x, abs_grad_y, dst;

	    //求x方向梯度
	    cv::Sobel(img, grad_x, -1, 1, 0, 3, 1, 1,cv::BORDER_DEFAULT);
	    cv::convertScaleAbs(grad_x, abs_grad_x);
	    cv::imshow("x方向soble", abs_grad_x);

	    //求y方向梯度
	    cv::Sobel(img, grad_y, -1, 0, 1,3, 1, 1, cv::BORDER_DEFAULT);
	    cv::convertScaleAbs(grad_y,abs_grad_y);
	    //cv::imshow("y向soble", abs_grad_y);

	    //合并梯度
	    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	    //cv::imshow("整体方向soble", dst);



		cv::imshow("sobel", dst);
		cv::waitKey(0);



	}

	return 0;
}
