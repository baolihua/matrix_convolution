/*
 *    Gx = [-1 0 1]
 *         [-2 0 2]
 *         [-1 0 1]

 *    Gy = [-1 -2 1]
 *         [ 0  0 0]
 *         [ 1  2 1]
 *
 */

typedef int data_type;

#define  ActNone        0
#define  ActRelu        1
#define  ActRelu1       2
#define  ActRelu6       3
#define  ActTanh        4
#define  ActSignBit     5
#define  ActSigmoid     6

#define PaddingSame 1
#define PaddingValid 2
 

__kernel void sobel(__global uchar* img_input, __global uchar* img_output, uint IMG_W, uint IMG_H)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int idx = 0;
    int threshold = 255;
    int  sum_h = 0;
    int sum_v = 0;
    int val = 0;

    idx = (y-1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;

    idx = (y-1)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

    idx = (y-1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

    idx = (y+1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    idx = (y+1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y)*IMG_W + (x);
    //int out = (int)fabs((sum_h/(float)9)*(float)0.5 + (sum_v/(float)9)*(float)0.5); 
    int out = (int)(fabs(sum_h/(float)9) * (float)0.5 + fabs(sum_h/(float)9) * (float)0.5);
    img_output[idx] = out > threshold ? 255 : out;
}

__kernel void sobel_matrix(__global data_type * img_input, __global data_type* img_output, __global uint *debug_out, uint IMG_W, uint IMG_H)
{
	int k=0;
	
	int i=get_global_id(0);
	int j=get_global_id(1);
	int width = get_global_size(0);
	
	int index = j*width + i;
  	data_type matrix[27] = {-1, 0, 1, -2, 0, 2, -1, 0, 1, -1, 0, 1, -2, 0, 2, -1, 0, 1, -1, 0, 1, -2, 0, 2, -1, 0, 1};

	//debug_out[index] = index;
	for(k=0; k<27; k++ ){
		img_output[index] +=  matrix[k] * img_input[index * 27 + k];
	}
}


__kernel void sobel_xyz(__global uchar* img_input, __global uchar* img_output, uint IMG_W, uint IMG_H)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int idx = 0;
    int threshold = 255;
    int  sum_h = 0;
    int sum_v = 0;
    int val = 0;
	uint img_chanel_size = IMG_W * IMG_H;
	
	/*r*/
    idx = (y-1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;

	/*g*/
	idx = (y-1)*IMG_W + (x-1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;

	/*b*/
	idx = (y-1)*IMG_W + (x-1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;
    /*----------------------------------------*/
	/*r*/
    idx = (y-1)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

	/*g*/
	idx = (y-1)*IMG_W + (x) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

	/*b*/
	idx = (y-1)*IMG_W + (x) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

	/*----------------------------------------*/
    /*r*/
    idx = (y-1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y-1)*IMG_W + (x+1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y-1)*IMG_W + (x+1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;
    /*------------------------------------------*/
	/*r*/
    idx = (y)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x-1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x-1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

	/*-------------------------------------------*/
    idx = (y)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 0;

    /*-------------------------------------------*/
    idx = (y)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x+1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x+1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

	/*-------------------------------------------*/
    idx = (y+1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x-1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x-1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

	/*--------------------------------------------*/
    idx = (y+1)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    idx = (y+1)*IMG_W + (x) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    idx = (y+1)*IMG_W + (x) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    /*--------------------------------------------*/
    idx = (y+1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x+1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x+1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y)*IMG_W + (x);
    //int out = (int)fabs((sum_h/(float)9)*(float)0.5 + (sum_v/(float)9)*(float)0.5); 
    int out = (int)(fabs(sum_h/(float)9) * (float)0.5 + fabs(sum_h/(float)9) * (float)0.5);
    img_output[idx] = out > threshold ? 255 : out;
}

/*
 *    Gx = [-1 0 1]
 *         [-2 0 2]
 *         [-1 0 1]
 */


__kernel void sobel_horizontal_rgb(global data_type *input, global data_type *filter, 
                       global data_type *bias, global data_type *output, global int* conv_params)
{
	int output_W = 0;
	int output_H = 0;
	int output_C = 0;

	int output_w = 0;
	int output_h = 0;
	int output_c = 0;

	int input_W = conv_params[0];
	int input_H = conv_params[1];

	int stride_w = conv_params[2]; 
	int stride_h = conv_params[3];
	int filter_w = conv_params[4];
	int filter_h = conv_params[5];
	int filter_c = conv_params[6];
	int active_func_type = conv_params[7];
	int padding = conv_params[8];
	int padding_top = conv_params[9];
	int padding_bottom = conv_params[10];
	int padding_left = conv_params[11];
	int padding_right = conv_params[12];
	int need_padding = 0;

	int input_index = 0;
	int input_w = 0;
	int input_h = 0;
	int output_index = 0;
	int filter_index = 0;
	int conv_result = 0;

	output_W = get_global_size(0);
	output_H = get_global_size(1);
	output_C = get_global_size(2);

	output_w = get_global_id(0);
	output_h = get_global_id(1);
	output_c = get_global_id(2);

	//output_index = output_c + output_w*output_C + output_h*output_W*output_C;
	output_index = output_w + output_h*output_W + output_c*output_W*output_H;
	input_w = output_w * stride_w;
	input_h = output_h * stride_h;
	data_type input_data = 0;
	//output[output_index] = output_index;
	
	for(int h_=0; h_ < filter_h; h_++)
		for(int w_=0; w_ < filter_w; w_++)
			for(int c_=0; c_ < filter_c; c_++){
				need_padding = 0;
				//input_index = c_ + (input_w + w_ - padding_left)*filter_c + (input_h + h_ - padding_bottom)*filter_c*input_W;
				input_index = c_ * input_W * input_H + (input_w + w_ - padding_left) + (input_h + h_ - padding_bottom)*input_W;

				//filter_index = c_ + w_*filter_c + h_*filter_c*filter_w + output_c * filter_c * filter_w *filter_h;
				
				filter_index = c_ * filter_w *filter_h + w_ + h_*filter_w + output_c * filter_c * filter_w *filter_h;
				
				if(padding == PaddingSame){
					if( ((input_w + w_ - padding_left) < 0) || ((input_w + w_ - padding_left) >= input_W) )
					  need_padding = 1;
					else if ( ((input_h + h_ - padding_bottom) < 0) || ((input_h + h_ - padding_bottom) >= input_H)  )
					  need_padding = 1;
				}
				if(need_padding == 0){
					
					conv_result += input[input_index] * filter[filter_index];
			
				}
			}
			//output[output_index] = conv_result;
			
			
			conv_result += bias[output_c];
			switch(active_func_type){
				case ActNone:
					output[output_index] = conv_result;
					break;
				case ActRelu:
					output[output_index] = conv_result > 0 ? conv_result : 0;
					break;
				case ActRelu1:
					conv_result = conv_result > 0 ? conv_result : 0; 
					output[output_index] = conv_result > 1 ? 1 : conv_result;
					break;
				case ActRelu6:
					conv_result = conv_result > 0 ? conv_result : 0; 
					output[output_index] = conv_result > 6 ? 6 : conv_result;
					break;
		      }
		  
		    
}


__kernel void sobel_horizontal_xyz(__global uchar* img_input, __global uchar* img_output, uint IMG_W, uint IMG_H)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int idx = 0;
    int threshold = 255;
    int  sum_h = 0;
    int sum_v = 0;
    int val = 0;
	uint img_chanel_size = IMG_W * IMG_H;
	
	/*r*/
    idx = (y-1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;

	/*g*/
	idx = (y-1)*IMG_W + (x-1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;

	/*b*/
	idx = (y-1)*IMG_W + (x-1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * -1;
    /*----------------------------------------*/
	/*r*/
    idx = (y-1)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

	/*g*/
	idx = (y-1)*IMG_W + (x) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

	/*b*/
	idx = (y-1)*IMG_W + (x) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * -2;

	/*----------------------------------------*/
    /*r*/
    idx = (y-1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y-1)*IMG_W + (x+1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y-1)*IMG_W + (x+1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;
    /*------------------------------------------*/
	/*r*/
    idx = (y)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x-1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x-1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * -2;
    sum_v += val * 0;

	/*-------------------------------------------*/
    idx = (y)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 0;

    /*-------------------------------------------*/
    idx = (y)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x+1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

    idx = (y)*IMG_W + (x+1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 2;
    sum_v += val * 0;

	/*-------------------------------------------*/
    idx = (y+1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x-1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x-1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * -1;
    sum_v += val * 1;

	/*--------------------------------------------*/
    idx = (y+1)*IMG_W + (x);
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    idx = (y+1)*IMG_W + (x) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    idx = (y+1)*IMG_W + (x) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 0;
    sum_v += val * 2;

    /*--------------------------------------------*/
    idx = (y+1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x+1) + img_chanel_size;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x+1) + img_chanel_size * 2;
    val = img_input[idx];
    sum_h += val * 1;
    sum_v += val * 1;

    idx = (y)*IMG_W + (x);
    //int out = (int)fabs((sum_h/(float)9)*(float)0.5 + (sum_v/(float)9)*(float)0.5); 
    int out = (int)(fabs(sum_h/(float)9) * (float)0.5 + fabs(sum_h/(float)9) * (float)0.5);
    img_output[idx] = out > threshold ? 255 : out;
}

 
__kernel void sobel_horizontal(__global uchar* img_input, __global uchar* img_output, uint IMG_W, uint IMG_H)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = 0;
 
    int sum_h = 0;
    int threshold = 255;

    idx = (y-1)*IMG_W + (x-1);
    sum_h += img_input[idx] * -1;

    idx = (y-1)*IMG_W + (x);
    sum_h += img_input[idx] * 0;

    idx = (y-1)*IMG_W + (x+1);
    sum_h += img_input[idx] * 1;

    idx = (y)*IMG_W + (x-1);
    sum_h += img_input[idx] * -2;

    idx = (y)*IMG_W + (x);
    sum_h += img_input[idx] * 0;

    idx = (y)*IMG_W + (x+1);
    sum_h += img_input[idx] * 2;

    idx = (y+1)*IMG_W + (x-1);
    sum_h += img_input[idx] * -1;

    idx = (y+1)*IMG_W + (x);
    sum_h += img_input[idx] * 0;

    idx = (y+1)*IMG_W + (x+1);
    sum_h += img_input[idx] * 1;

    idx = (y)*IMG_W + (x);
    int out = (int)fabs(sum_h/(float)9);
    img_output[idx] = out > threshold ? 255 : out;
}

__kernel void sobel_vertical(__global uchar* img_input, __global uchar* img_output, uint IMG_W, uint IMG_H)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int idx = 0;
    int sum_v = 0;
    int val = 0;
    int threshold = 255;


    idx = (y-1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_v += val * -1;

    idx = (y-1)*IMG_W + (x);
    val = img_input[idx];
    sum_v += val * -2;

    idx = (y-1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_v += val * 1;

    idx = (y)*IMG_W + (x-1);
    val = img_input[idx];
    sum_v += val * 0;

    idx = (y)*IMG_W + (x);
    val = img_input[idx];
    sum_v += val * 0;

    idx = (y)*IMG_W + (x+1);
    val = img_input[idx];
    sum_v += val * 0;

    idx = (y+1)*IMG_W + (x-1);
    val = img_input[idx];
    sum_v += val * 1;

    idx = (y+1)*IMG_W + (x);
    val = img_input[idx];
    sum_v += val * 2;

    idx = (y+1)*IMG_W + (x+1);
    val = img_input[idx];
    sum_v += val * 1;

    idx = (y)*IMG_W + (x);
    int out = (int)fabs(sum_v/(float)9);
    img_output[idx] = out > threshold ? 255 : out;
}


/*

*	 Gx = [-1 0 1]
*		  [-2 0 2]
*		  [-1 0 1]

*	 Gy = [-1 -2 1]
*		  [ 0  0 0]
*		  [ 1  2 1]

*    G = sqrt(Gx*Gx + Gy*Gy)
*
*/

const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void sobel_rgb(read_only image2d_t src, write_only image2d_t dst)
{
	int x = (int)get_global_id(0);
	int y = (int)get_global_id(1);

	if(x >= get_image_width(src) || y >= get_image_height(src))
		return;

	float4 p00 = read_imagef(src, sampler, (int2)(x-1, y-1));
	float4 p10 = read_imagef(src, sampler, (int2)(x, y-1));
	float4 p20 = read_imagef(src, sampler, (int2)(x+1, y-1));

	float4 p01 = read_imagef(src, sampler, (int2)(x-1, y));
	float4 p21 = read_imagef(src, sampler, (int2)(x+1, y));

	float4 p02 = read_imagef(src, sampler, (int2)(x-1, y+1));
	float4 p12 = read_imagef(src, sampler, (int2)(x, y+1));

	float4 p22 = read_imagef(src, sampler, (int2)(x+1, y+1));

	float3 gx = -p00.xyz + p20.xyz + 2.0f * (p21.xyz - p01.xyz) - p02.xyz + p22.xyz;
	float3 gy = -p00.xyz - p20.xyz + 2.0f * (p12.xyz - p10.xyz) + p02.xyz + p22.xyz;
	float3 g = native_sqrt(gx * gx + gy * gy);

	write_imagef(dst, (int2)(x, y), (float4)(g.x, g.y, g.z, 1.0f));
}


