
#include "Image2matrix.h"

/*
**  从输入的多通道数组im（存储图像数据）中获取指定行、列、、通道数处的元素值
**  输入： im      输入，所有数据存成一个一维数组，例如对于3通道的二维图像而言，
**                每一通道按行存储（每一通道所有行并成一行），三通道依次再并成一行
**        height  每一通道的高度（即输入图像的真正的高度，补0之前）
**        width   每一通道的宽度（即输入图像的宽度，补0之前）
**        channels 输入im的通道数，比如彩色图为3通道，之后每一卷积层的输入的通道数等于上一卷积层卷积核的个数
**        row     要提取的元素所在的行（二维图像补0之后的行数）
**        col     要提取的元素所在的列（二维图像补0之后的列数）
**        channel 要提取的元素所在的通道
**        pad     图像左右上下各补0的长度（四边补0的长度一样）
**  返回： float类型数据，为im中channel通道，row-pad行，col-pad列处的元素值
**  注意：在im中并没有存储补0的元素值，因此height，width都是没有补0时输入图像真正的
**       高、宽；而row与col则是补0之后，元素所在的行列，因此，要准确获取在im中的元素值，
**       首先需要减去pad以获取在im中真实的行列数
*/
float im2col_get_pixel(data_type * im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    // 减去补0长度，获取元素真实的行列数
    row -= pad;
    col -= pad;

    // 如果行列数小于0,则返回0（刚好是补0的效果）
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;

    // im存储多通道二维图像的数据的格式为：各通道所有行并成一行，再多通道依次并成一行，
    // 因此width*height*channel首先移位到所在通道的起点位置，加上width*row移位到
    // 所在指定通道所在行，再加上col移位到所在列
    // im[col + width*(row + height*channel)]=im[col+width*row+width*height*channel]
    return im[col + width*(row + height*channel)];
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*
 将输入图片转为便于计算的数组格式，可以参考https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
 进行辅助理解（但执行方式并不同，只是用于概念上的辅助理解），由作者的注释可知，这是直接从caffe移植过来的
 输入： data_im    输入图像
       channels   输入图像的通道数（对于第一层，一般是颜色图，3通道，中间层通道数为上一层卷积核个数）
       height     输入图像的高度（行）
       width      输入图像的宽度（列）
       ksize      卷积核尺寸
       stride     卷积核跨度
       pad        四周补0长度
       data_col   相当于输出，为进行格式重排后的输入图像数据

注:
   data_col还是按行排列，
       行数为channels*ksize*ksize,
       列数为height_col*width_col，即一张特征图总的元素个数，

*/
#if 0
void im2col_cpu(unsigned char* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, unsigned char* data_col) 
{
    int c,h,w;
    // 卷积后的尺寸计算,这里width_col=width
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    /// 卷积核大小：ksize*ksize是一个卷积核的大小，通道数channels
    int channels_col = channels * ksize * ksize;

  // 获取channels_col个对应像素核

    printf("height_col = %d, width_col = %d, channels_col = %d\n", height_col, width_col, channels_col);
  
    for (c = 0; c < channels_col; ++c) {
        // 卷积核上的坐标:(w_offset,h_offset)
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;

        int c_im = c / ksize / ksize;
		printf("c = %d\n", c);
        for (h = 0; h < height_col; ++h) {
            // 内循环等于该层输出图像列数width_col，说明最终得到的data_col总有channels_col行，height_col*width_col列
            for (w = 0; w < width_col; ++w) {

                // 获取输入图像的对应像素坐标
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;

                // col_index为重排后图像中的像素索引，等于c * height_col * width_col + h * width_col +w
                int col_index = (c * height_col + h) * width_col + w;

                data_col[col_index] =  im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
		
    }
}

#endif

void im2col_cpu(data_type * data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, data_type * data_col) 
{
    int c,h,w;
    // 卷积后的尺寸计算,这里width_col=width
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    /// 卷积核大小：ksize*ksize是一个卷积核的大小，通道数channels
    int channels_col = channels * ksize * ksize;

    for (h = 0; h < height_col; ++h) {
        // 内循环等于该层输出图像列数width_col，说明最终得到的data_col总有channels_col行，height_col*width_col列
        for (w = 0; w < width_col; ++w) {
			
			for (c = 0; c < channels_col; ++c) {
				// 卷积核上的坐标:(w_offset,h_offset)
				int w_offset = c % ksize;
				int h_offset = (c / ksize) % ksize;
			
				int c_im = c / ksize / ksize;
				
				//printf("c = %d\n", c);

	            // 获取输入图像的对应像素坐标
	            int im_row = h_offset + h * stride;
	            int im_col = w_offset + w * stride;
				
				//printf("im_row = %d, im_col = %d, c_im = %d\n", im_row, im_col, c_im);

				// col_index为重排后图像中的像素索引，等于c * height_col * width_col + h * width_col +w
	            int col_index = c + w * channels_col + h * width_col * channels_col;
				//printf("col_index = %d\n", col_index);
	            data_col[col_index] =  im2col_get_pixel(data_im, height, width, channels,
	                    im_row, im_col, c_im, pad);
			 }
        }
    }
}



int MatixMulty(unsigned char* data_b, int width_b, unsigned char* data_a,int height_a,  int width_a)
{
	int i =0;
	int j;
	int k =0;
	unsigned char C[width_a] ={0};

	if(width_b != height_a){
		printf("%s, %d, error!\n", __FUNCTION__, __LINE__);
		return -1;
	}
	printf("%s, %d\n", __FUNCTION__, __LINE__);
	for(i=0; i<width_a; i++){
		j= i;
		for(; j<height_a * width_a; j+=width_a){
			//printf("%s, %d, data_b[%d] = %f, data_a[%d] = %f\n", __FUNCTION__, __LINE__, k, data_b[k], j, data_a[j]);
			C[i] +=  data_b[k] * data_a[j];
			k++;
		}
		k=0;
	}
	
	printf("%s, %d\n", __FUNCTION__, __LINE__);
	for(i = 0; i <= width_a - 1; i++){
			printf("C[%d] = %f\n", i, C[i]);
	}
	
	return 0;
}



float matix[4][4][3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
                    	33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0};


#if 0
int main()
{
	int i,j;
	unsigned char matix[4][4] = {1, 2, 3, 4,
		                 5, 6, 7, 8,
		                 9, 10, 11, 12,
		                 13, 14, 15, 16};
	unsigned char kernel[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	unsigned char data[9 * 16] = {0}; 
	im2col_cpu(&matix[0][0] , 1, 4, 4, 3, 1, 1, &data[0]);

	printf("\n");
	for(i=0; i<9 * 16; i++){
		if(i % 16 == 0){
			printf("\n");
		}
		printf("%f ", data[i]);
	}


	printf("----------------------------------\n");
	
	MatixMulty(&kernel[0], 9, data, 9, 16);
	return 0;
}
#endif


