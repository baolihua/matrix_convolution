#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>


#define MAX_SOURCE_SIZE (0x10000)

typedef int data_type;

const char* image_files[] = {
    //"./gray/12.jpg", NULL
	"./gray/13.jpg", NULL
};
struct iinfo {
    cv::Mat _img;
    unsigned char* _buf_out;
    cl_mem _cl_buf_in;
    cl_mem _cl_buf_out;
    void * map_buf_in;
    void * map_buf_out;
};
const int IMAGE_NUM = 1;
struct iinfo image_infos[IMAGE_NUM];

#define CHECK_OPENCL_ERROR(status, str) \
	if (status != CL_SUCCESS) printf("%s\n", str)


static int timespec_check(struct timespec *t)
{
	if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
		return -1;
	return 0;

}

void timespec_sub(struct timespec *t1, struct timespec *t2)
{
	if (timespec_check(t1) < 0) {
		printf("invalid time #1: %lld.%.9ld.\n",
			(long long)t1->tv_sec, t1->tv_nsec);
		return;
	}
	if (timespec_check(t2) < 0) {
		printf("invalid time #2: %lld.%.9ld.\n",
			(long long)t2->tv_sec, t2->tv_nsec);
		return;
	}
	t1->tv_sec -= t2->tv_sec;
	t1->tv_nsec -= t2->tv_nsec;
	if (t1->tv_nsec >= 1000000000) {
		t1->tv_sec++;
		t1->tv_nsec -= 1000000000;
	} else if (t1->tv_nsec < 0) {
		t1->tv_sec--;
		t1->tv_nsec += 1000000000;
	}
}

int save_data(unsigned char* buffer, unsigned int size)
{
	unsigned int count=0;
	FILE *p = fopen("matrix_b.txt", "w+");
	if(!p){
		printf("open file failed!\n");
		return -1;
	}
	if(!buffer){
		printf("buffer is NULL!\n");
		return -1;
	}
    for(count=0; count<size; count++){
		fprintf(p, "%d\n", *(buffer+count));
    }
	fclose(p);
	return 0;
}

int show_data(int * buffer, unsigned int row, unsigned int col)
{
	int i;
	for(i=0; i<row*col; i++){
		if(i % col == 0){
			printf("\n");
		}
		printf("%d ", buffer[i]); 
	}
	printf("\n");
	return 0;
}




/**获取编译program出错时，编译器的出错信息*/
int getProgramBuildInfo(cl_program program,cl_device_id device)
{
    size_t log_size;
    char *program_log;
    /* Find size of log and print to std output */
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
    program_log = (char*) malloc(log_size+1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size+1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    return 0;
}

int main()
{
    // Load the kernel source code into the array source_str
	ssize_t rc;
	struct timespec ts_start, ts_end;
	struct timespec ts_start0, ts_end0;
	
	long total_time = 0;
	float avg_time = 0;

	long total_time0 = 0;
	float avg_time0 = 0;
	
    FILE *fp;
    char *source_str;
    size_t source_size;
	unsigned int image_size;
	unsigned int memsize =0;
	
	//unsigned int in_image_size=0;
	//unsigned int out_memsize =0;
	
	int height_col;
	int width_col;
	int pad = 0;
	int ksize = 3;
	int stride = 1;
	int channels_col;

	data_type kernel_sobel_rgb[3][3][3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1, -1, 0, 1, -2, 0, 2, -1, 0, 1, -1, 0, 1, -2, 0, 2, -1, 0, 1};
	int conv_params[13];

	data_type *input_data = NULL;
	unsigned int input_data_size;

	unsigned char *output_data = NULL;
	unsigned int cl_output_data_size;
	unsigned int count;
	unsigned int cnt;
	unsigned int w,h,c;
	unsigned int index;
	unsigned int buffer_temp[27] = {0};

	cl_mem cl_kernel_sobel_rgb;
	void *map_cl_kernel_sobel_rgb = NULL;
	
	cl_mem cl_bias;
	void *map_cl_bias = NULL;
	
	cl_mem cl_conv_params;
	void *map_cl_conv_params = NULL;

	cl_mem cl_output;
	void *map_cl_output = NULL;
 
    fp = fopen("sobel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );


    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CHECK_OPENCL_ERROR(ret, "clGetPlatformIDs() failed");
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);
    CHECK_OPENCL_ERROR(ret, "clGetDeviceIDs() failed");
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    CHECK_OPENCL_ERROR(ret, "clCreateContext() failed");
  
    // Create a command queue
    //cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    CHECK_OPENCL_ERROR(ret, "clCreateCommandQueue() failed");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    CHECK_OPENCL_ERROR(ret, "clBuildProgram() failed");
    if (ret != CL_SUCCESS)    
	getProgramBuildInfo(program, device_id);
 
    // Create the OpenCL kernel
    //cl_kernel kernel = clCreateKernel(program, "sobel_xyz", &ret);
    //cl_kernel kernel = clCreateKernel(program, "sobel_vertical", &ret);
    cl_kernel kernel = clCreateKernel(program, "sobel_horizontal_rgb", &ret);
    CHECK_OPENCL_ERROR(ret, "clCreateKernel() failed");
    size_t max_dim;
    size_t max_item_size[3] = {1, 1, 1};
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &max_dim, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, max_item_size, NULL);
    printf("max_dim = %d, [%d, %d, %d]\n", (int)max_dim, (int)max_item_size[0], (int)max_item_size[1], (int)max_item_size[2]);

    for (int i=0; i<IMAGE_NUM; i++) {
    	//image_infos[i]._img = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
    	image_infos[i]._img = cv::imread(image_files[i], cv::IMREAD_UNCHANGED);
        if (image_infos[i]._img.data == NULL) {
		    printf("failed to read image %s\n", image_files[i]);
		    return -1;
        }
			
		height_col = (image_infos[i]._img.rows + 2*pad - ksize) / stride + 1;
		width_col = (image_infos[i]._img.cols + 2*pad - ksize) / stride + 1;
		channels_col = image_infos[i]._img.channels();
		memsize = height_col * width_col;
#if 0
		w = 0;
		for(c=0; c<channels_col; c++){
			for(h=0; h<ksize; h++){
				for(cnt=0; cnt<ksize; cnt++){
					index = cnt + h * image_infos[i]._img.cols +  c * image_infos[i]._img.cols * image_infos[i]._img.rows;
					buffer_temp[w++] = image_infos[i]._img.data[index];
				}
			}
		}

		printf("\n");
		
		for(w=0; w<27; w++){
			printf("%d ", buffer_temp[w]);
		}

		printf("\n");


		for(c=0; c<channels_col; c++){
			printf("\n-------------------------------------------------------------------------\n");
			for(h=0; h<ksize; h++){
				printf("\n===================================\n");
				for(cnt=0; cnt<image_infos[i]._img.cols; cnt++){
					index = cnt + h*image_infos[i]._img.cols + c * image_infos[i]._img.cols * image_infos[i]._img.rows;
					printf("%d ", image_infos[i]._img.data[index]);
				}
			}
		}
#endif
		/*char*/
    	image_size = image_infos[i]._img.cols * image_infos[i]._img.rows * image_infos[i]._img.channels();

		/*int */
		input_data_size = image_size * sizeof(data_type);

		/*int*/
		cl_output_data_size = memsize *sizeof(data_type);
		
		output_data = (unsigned char *)malloc(memsize);
		if(!output_data){
            printf("malloc output data buffer failed, return\n");
            return -1;
		}
		memset(output_data, 0, memsize);

        // Create memory buffers on the device for each vector 
        image_infos[i]._cl_buf_in  = clCreateBuffer(context, CL_MEM_READ_WRITE, input_data_size, NULL, &ret);
        image_infos[i]._cl_buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_output_data_size, NULL, &ret);
		
	    image_infos[i].map_buf_in =clEnqueueMapBuffer(command_queue, image_infos[i]._cl_buf_in ,CL_TRUE,CL_MAP_WRITE, 0, input_data_size,0, NULL,NULL,&ret);

		/*char data change to int data_type data*/
		for(count=0; count<image_size; count++){
			((data_type *)image_infos[i].map_buf_in)[count] = (data_type)image_infos[i]._img.data[count];
		}

		cl_kernel_sobel_rgb = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(kernel_sobel_rgb), NULL, &ret);
		cl_conv_params = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(conv_params), NULL, &ret);
		cl_bias = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_output_data_size, NULL, &ret);

		map_cl_kernel_sobel_rgb = clEnqueueMapBuffer(command_queue, cl_kernel_sobel_rgb,CL_TRUE,CL_MAP_WRITE, 0, sizeof(kernel_sobel_rgb),0, NULL,NULL,&ret);
		map_cl_conv_params = clEnqueueMapBuffer(command_queue, cl_conv_params,CL_TRUE,CL_MAP_WRITE, 0, sizeof(conv_params),0, NULL,NULL,&ret);
		map_cl_bias = clEnqueueMapBuffer(command_queue, cl_bias,CL_TRUE,CL_MAP_WRITE, 0, cl_output_data_size,0, NULL,NULL,&ret);

    }

    int loops = 1;
	rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
	if (rc < 0){
		printf("get start time error!\n");
		return -1;
	}
    while(loops-- > 0) {
        for (int i=0; i<IMAGE_NUM; i++) {
            //cv::imshow("origin", image_infos[i]._img);
            // Set the arguments of the kernel
	        int width  = image_infos[i]._img.cols;
	        int height = image_infos[i]._img.rows;
			
			/*input w * h */
			conv_params[0] = width;
			conv_params[1] = height;

			/*stripe w * h*/
			conv_params[2] = 1;
			conv_params[3] = 1;

			/*kernel w *h *c */
			conv_params[4] = 3;
			conv_params[5] = 3;
			conv_params[6] = 3;

			conv_params[7] = 0;

			/*padding*/
			conv_params[8] = 2;
			conv_params[9] = 0;
			conv_params[10] = 0;
			conv_params[11] = 0;
			conv_params[12] = 0;
			
			clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_in,image_infos[i].map_buf_in,0,NULL,NULL);

			memcpy(map_cl_kernel_sobel_rgb, kernel_sobel_rgb, sizeof(kernel_sobel_rgb));
			memcpy(map_cl_conv_params, conv_params, sizeof(conv_params));
			memset(map_cl_bias, 0, cl_output_data_size);

			clEnqueueUnmapMemObject(command_queue,cl_kernel_sobel_rgb, map_cl_kernel_sobel_rgb,0,NULL,NULL);
			clEnqueueUnmapMemObject(command_queue,cl_conv_params, map_cl_conv_params,0,NULL,NULL);
			clEnqueueUnmapMemObject(command_queue,cl_bias, map_cl_bias,0,NULL,NULL);

			rc = clock_gettime(CLOCK_MONOTONIC, &ts_start0);
			if (rc < 0){
				printf("get start time error!\n");
				return -1;
			}
			
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_infos[i]._cl_buf_in);
            ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&image_infos[i]._cl_buf_out);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_kernel_sobel_rgb);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cl_bias);
            ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cl_conv_params);
			
            CHECK_OPENCL_ERROR(ret, "clSetKernelArg() failed");
 

            printf("w:%d, h:%d, c:%d\n", image_infos[i]._img.cols, image_infos[i]._img.rows, image_infos[i]._img.channels());
            // Execute the OpenCL kernel on the list
            
            //size_t global_item_size[3] = {(size_t)image_infos[i]._img.cols, (size_t)image_infos[i]._img.rows,3}; // Process the entire lists
            size_t global_item_size[3] = {(size_t)width_col, (size_t)height_col,1}; // Process the entire lists
			size_t local_item_size[3] = {1, 1, 1}; // Divide work items into groups of 64
            size_t global_item_offset[3] = {0, 0, 0};

			ret = clEnqueueNDRangeKernel(command_queue, kernel, sizeof(global_item_size)/sizeof(global_item_size[0]), global_item_offset, 
                                         global_item_size, local_item_size, 0, NULL, NULL); 
            CHECK_OPENCL_ERROR(ret, "clEnqueueNDRangeKernel() failed");

	        clFlush(command_queue);
    	    clFinish(command_queue);

			rc = clock_gettime(CLOCK_MONOTONIC, &ts_end0);
			if (rc < 0){
				printf("get start time error!\n");
				return -1;
			}
			
            //cv::Mat dst(image_infos[i]._img.rows, image_infos[i]._img.cols, CV_8UC1, (void*)image_infos[i].map_buf_out);
            //cv::imshow("processed", dst);

	        image_infos[i].map_buf_out= clEnqueueMapBuffer(command_queue, image_infos[i]._cl_buf_out ,CL_TRUE,CL_MAP_READ, 0, cl_output_data_size,0, NULL,NULL,&ret);
            CHECK_OPENCL_ERROR(ret, "map_buf_out failed");
			
			/*int to char*/
			for(count=0; count<memsize; count++){
				output_data[count] = (unsigned char)(((data_type *)image_infos[i].map_buf_out)[count]);
			}
			
			//show_data((int *)image_infos[i].map_buf_out, width_col, height_col);
			save_data(output_data, memsize);
            if (cv::waitKey(500) == 'q')
            	break; 
        }
    }
	rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);
	if (rc < 0){
		printf("get end time error!\n");
		return -1;
	}

	timespec_sub(&ts_end0, &ts_start0);
	total_time0 += ts_end0.tv_nsec;
	avg_time0 = (float)total_time0/1.0;
	printf("avg time 0: %f sec, %f nsec\n", ts_end0.tv_sec/1.0, avg_time0);
	
	timespec_sub(&ts_end, &ts_start);
	total_time += ts_end.tv_nsec;
	avg_time = (float)total_time/1.0;
	printf("avg time: %f sec, %f nsec\n", ts_end.tv_sec/1.0, avg_time);
	
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    for (int i=0; i<IMAGE_NUM; i++) {
		//free(image_infos[i]._buf_out);
		free(output_data);
		clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_in,image_infos[i].map_buf_in,0,NULL,NULL);
		clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_out,image_infos[i].map_buf_out,0,NULL,NULL);

		ret = clReleaseMemObject(image_infos[i]._cl_buf_in);
		ret = clReleaseMemObject(image_infos[i]._cl_buf_out);
    }
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);   
    return 0;
}

