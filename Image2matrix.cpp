
#include "Image2matrix.h"

/*
**  ������Ķ�ͨ������im���洢ͼ�����ݣ��л�ȡָ���С��С���ͨ��������Ԫ��ֵ
**  ���룺 im      ���룬�������ݴ��һ��һά���飬�������3ͨ���Ķ�άͼ����ԣ�
**                ÿһͨ�����д洢��ÿһͨ�������в���һ�У�����ͨ�������ٲ���һ��
**        height  ÿһͨ���ĸ߶ȣ�������ͼ��������ĸ߶ȣ���0֮ǰ��
**        width   ÿһͨ���Ŀ�ȣ�������ͼ��Ŀ�ȣ���0֮ǰ��
**        channels ����im��ͨ�����������ɫͼΪ3ͨ����֮��ÿһ�����������ͨ����������һ��������˵ĸ���
**        row     Ҫ��ȡ��Ԫ�����ڵ��У���άͼ��0֮���������
**        col     Ҫ��ȡ��Ԫ�����ڵ��У���άͼ��0֮���������
**        channel Ҫ��ȡ��Ԫ�����ڵ�ͨ��
**        pad     ͼ���������¸���0�ĳ��ȣ��ı߲�0�ĳ���һ����
**  ���أ� float�������ݣ�Ϊim��channelͨ����row-pad�У�col-pad�д���Ԫ��ֵ
**  ע�⣺��im�в�û�д洢��0��Ԫ��ֵ�����height��width����û�в�0ʱ����ͼ��������
**       �ߡ�����row��col���ǲ�0֮��Ԫ�����ڵ����У���ˣ�Ҫ׼ȷ��ȡ��im�е�Ԫ��ֵ��
**       ������Ҫ��ȥpad�Ի�ȡ��im����ʵ��������
*/
float im2col_get_pixel(data_type * im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    // ��ȥ��0���ȣ���ȡԪ����ʵ��������
    row -= pad;
    col -= pad;

    // ���������С��0,�򷵻�0���պ��ǲ�0��Ч����
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;

    // im�洢��ͨ����άͼ������ݵĸ�ʽΪ����ͨ�������в���һ�У��ٶ�ͨ�����β���һ�У�
    // ���width*height*channel������λ������ͨ�������λ�ã�����width*row��λ��
    // ����ָ��ͨ�������У��ټ���col��λ��������
    // im[col + width*(row + height*channel)]=im[col+width*row+width*height*channel]
    return im[col + width*(row + height*channel)];
}


//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*
 ������ͼƬתΪ���ڼ���������ʽ�����Բο�https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
 ���и�����⣨��ִ�з�ʽ����ͬ��ֻ�����ڸ����ϵĸ�����⣩�������ߵ�ע�Ϳ�֪������ֱ�Ӵ�caffe��ֲ������
 ���룺 data_im    ����ͼ��
       channels   ����ͼ���ͨ���������ڵ�һ�㣬һ������ɫͼ��3ͨ�����м��ͨ����Ϊ��һ�����˸�����
       height     ����ͼ��ĸ߶ȣ��У�
       width      ����ͼ��Ŀ�ȣ��У�
       ksize      ����˳ߴ�
       stride     ����˿��
       pad        ���ܲ�0����
       data_col   �൱�������Ϊ���и�ʽ���ź������ͼ������

ע:
   data_col���ǰ������У�
       ����Ϊchannels*ksize*ksize,
       ����Ϊheight_col*width_col����һ������ͼ�ܵ�Ԫ�ظ�����

*/
#if 0
void im2col_cpu(unsigned char* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, unsigned char* data_col) 
{
    int c,h,w;
    // �����ĳߴ����,����width_col=width
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    /// ����˴�С��ksize*ksize��һ������˵Ĵ�С��ͨ����channels
    int channels_col = channels * ksize * ksize;

  // ��ȡchannels_col����Ӧ���غ�

    printf("height_col = %d, width_col = %d, channels_col = %d\n", height_col, width_col, channels_col);
  
    for (c = 0; c < channels_col; ++c) {
        // ������ϵ�����:(w_offset,h_offset)
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;

        int c_im = c / ksize / ksize;
		printf("c = %d\n", c);
        for (h = 0; h < height_col; ++h) {
            // ��ѭ�����ڸò����ͼ������width_col��˵�����յõ���data_col����channels_col�У�height_col*width_col��
            for (w = 0; w < width_col; ++w) {

                // ��ȡ����ͼ��Ķ�Ӧ��������
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;

                // col_indexΪ���ź�ͼ���е���������������c * height_col * width_col + h * width_col +w
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
    // �����ĳߴ����,����width_col=width
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    /// ����˴�С��ksize*ksize��һ������˵Ĵ�С��ͨ����channels
    int channels_col = channels * ksize * ksize;

    for (h = 0; h < height_col; ++h) {
        // ��ѭ�����ڸò����ͼ������width_col��˵�����յõ���data_col����channels_col�У�height_col*width_col��
        for (w = 0; w < width_col; ++w) {
			
			for (c = 0; c < channels_col; ++c) {
				// ������ϵ�����:(w_offset,h_offset)
				int w_offset = c % ksize;
				int h_offset = (c / ksize) % ksize;
			
				int c_im = c / ksize / ksize;
				
				//printf("c = %d\n", c);

	            // ��ȡ����ͼ��Ķ�Ӧ��������
	            int im_row = h_offset + h * stride;
	            int im_col = w_offset + w * stride;
				
				//printf("im_row = %d, im_col = %d, c_im = %d\n", im_row, im_col, c_im);

				// col_indexΪ���ź�ͼ���е���������������c * height_col * width_col + h * width_col +w
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


