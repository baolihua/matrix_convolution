#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>
#include "Image2matrix.h"


#define MAX_SOURCE_SIZE (0x10000)

typedef int data_type;

const char* image_files[] = {
    "./gray/13.jpg", NULL
};
struct iinfo {
    cv::Mat _img;
    signed char* _buf_out;
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


int save_data(unsigned char * buffer, unsigned int size)
{
	unsigned int count=0;
	FILE *p = fopen("matrix_a.txt", "w+");
	if(!p){
		printf("open file failed!\n");
		return -1;
	}
	if(!buffer){
		printf("buffer is NULL!\n");
		return -1;
	}
/*
	count = fwrite(buffer, sizeof(unsigned char), size, p);
	if(count != size){
		printf("count = %d, size = %d\n", count, size);
		fclose(p);
		return -1;
	}
*/
    for(count=0; count<size; count++){
		fprintf(p, "%d\n", *(buffer+count));
    }
	fclose(p);
	return 0;
}

int show_data(unsigned int * buffer, unsigned int size)
{
	int i;
	printf("\n");
	for(i=0; i<size; i++){
		printf("buffer[%d] = %d\n", i, buffer[i]);
	}
	printf("\n");
	return 0;
}

int show_result(data_type * buffer, unsigned int size)
{
	int i;
	printf("\n");
	for(i=0; i<size; i++){
		printf("buffer[%d] = %d\n", i, buffer[i]);
	}
	printf("\n");
	return 0;
}

int show_matrix_result(data_type * buffer, unsigned int size)
{
	int i;
	printf("\n");
	for(i=0; i<size; i++){
		if(i % 27 == 0){
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
	struct timespec ts_start1, ts_end1;
	long total_time = 0;
	float avg_time = 0;

	long total_time0 = 0;
	float avg_time0 = 0;

	long total_time1 = 0;
	float avg_time1 = 0;
	
    FILE *fp;
    char *source_str;
    size_t source_size;
    //unsigned char* pchanged_data = NULL;
    unsigned int in_image_size;
	unsigned int in_matrix_size;
	unsigned int out_memsize = 0;
	int height_col;
	int width_col;
	int channels_col;
	int pad = 0;
	int ksize = 3;
	int stride = 1;
	int count;
	int output_data_count;
	int cnt;

	int *buffer_temp = NULL;

	cl_mem cl_debug_out;
	void *map_cl_debug_out = NULL;

	int *image_int_data = NULL;
	unsigned char *image_output_char_data = NULL;

	data_type *output_data = NULL;
    
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
    cl_kernel kernel = clCreateKernel(program, "sobel_matrix", &ret);
    CHECK_OPENCL_ERROR(ret, "clCreateKernel() failed");
    size_t max_dim;
    size_t max_item_size[3] = {1, 1, 1};
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &max_dim, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, max_item_size, NULL);
    printf("max_dim = %d, [%d, %d, %d]\n", (int)max_dim, (int)max_item_size[0], (int)max_item_size[1], (int)max_item_size[2]);

    for (int i=0; i<IMAGE_NUM; i++) {
		
    	image_infos[i]._img = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
    	image_infos[i]._img = cv::imread(image_files[i], cv::IMREAD_UNCHANGED);
        if (image_infos[i]._img.data == NULL) {
		    printf("failed to read image %s\n", image_files[i]);
		    return -1;
        }
		height_col = (image_infos[i]._img.rows + 2*pad - ksize) / stride + 1;
		width_col = (image_infos[i]._img.cols + 2*pad - ksize) / stride + 1;
		channels_col = image_infos[i]._img.channels();

		output_data_count= height_col * width_col;

		/*byte: unsigned char*/
		in_image_size = image_infos[i]._img.cols * image_infos[i]._img.rows * image_infos[i]._img.channels();

		/*data_type*/
		in_matrix_size = height_col * width_col * channels_col * ksize * ksize * sizeof(data_type);
		out_memsize = height_col * width_col * sizeof(data_type);

		/*unsigned char to int*/
		image_int_data = (int *)malloc(in_image_size * sizeof(data_type));
		if(!image_int_data){
			printf("image data is NULL!\n");
			return -1;
		}

		/*int to unsigned char*/
		image_output_char_data = (unsigned char *)malloc(in_image_size);
		if(!image_output_char_data){
			printf("image output data buffer is NULL!\n");
			return -1;
		}

		output_data = (data_type *)malloc(out_memsize);
		if(!output_data){
			printf("output data buffer is NULL!\n");
			return -1;
		}

		buffer_temp = (int *)malloc(in_matrix_size);
		if(!buffer_temp){
			printf("buffer_temp buffer is NULL!\n");
			return -1;
		}

		/*change the image char data to int*/
		for(count=0; count<in_image_size; count++){
			image_int_data[count] = image_infos[i]._img.data[count];
		}

        //image_infos[i]._buf_out = (unsigned char*)malloc(out_memsize);
        //if (!image_infos[i]._buf_out) {
        //    printf("malloc image buffer failed, return\n");
        //    return -1;
        //}
		
        // Create memory buffers on the device for each vector 
        image_infos[i]._cl_buf_in  = clCreateBuffer(context, CL_MEM_READ_WRITE, in_matrix_size, NULL, &ret);
        image_infos[i]._cl_buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, out_memsize, NULL, &ret);

		/*debug*/
		cl_debug_out = clCreateBuffer(context, CL_MEM_READ_WRITE, (out_memsize + 2), NULL, &ret);
		map_cl_debug_out = clEnqueueMapBuffer(command_queue, cl_debug_out,CL_TRUE,CL_MAP_WRITE, 0, (out_memsize + 2),0, NULL,NULL,&ret);
		memset(map_cl_debug_out, 0, out_memsize);
		
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

			unsigned int imgwidth = image_infos[i]._img.cols;
			unsigned int imgheigh = image_infos[i]._img.rows;

            // Set the arguments of the kernel
	        cl_uint width  = ksize * ksize * channels_col * width_col;
	        cl_uint height = height_col;
			
			image_infos[i].map_buf_in=clEnqueueMapBuffer(command_queue, image_infos[i]._cl_buf_in ,CL_TRUE,CL_MAP_WRITE, 0, in_matrix_size,0, NULL,NULL,&ret);

			/*test image to matrix time */
			rc = clock_gettime(CLOCK_MONOTONIC, &ts_start0);
			if (rc < 0){
				printf("get start time error!\n");
				return -1;
			}
			
			//im2col_cpu(image_int_data, image_infos[i]._img.channels(), imgheigh, imgwidth, 3, 1, 0, (data_type *)image_infos[i].map_buf_in);
			im2col_cpu(image_int_data, image_infos[i]._img.channels(), imgheigh, imgwidth, 3, 1, 0, (data_type *)buffer_temp);
			memcpy(image_infos[i].map_buf_in, buffer_temp, in_matrix_size);
			
			rc = clock_gettime(CLOCK_MONOTONIC, &ts_end0);
			if (rc < 0){
				printf("get end time error!\n");
				return -1;
			}
			
		    clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_in,image_infos[i].map_buf_in,0,NULL,NULL);


			/*test kernel time*/
			rc = clock_gettime(CLOCK_MONOTONIC, &ts_start1);
			if (rc < 0){
				printf("get start time error!\n");
				return -1;
			}
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_infos[i]._cl_buf_in);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_infos[i]._cl_buf_out);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cl_debug_out);
            ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&width);
            ret = clSetKernelArg(kernel, 4, sizeof(cl_uint), (void *)&height);
            CHECK_OPENCL_ERROR(ret, "clSetKernelArg() failed");
 
            //printf("w:%d, h:%d, c:%d\n", image_infos[i]._img.cols, image_infos[i]._img.rows, image_infos[i]._img.channels());
			size_t global_item_size[2] = {(size_t)(width_col), (size_t)height_col}; // Process the entire lists
			size_t local_item_size[2] = {1, 1}; // Divide work items into groups of 64
            size_t global_item_offset[2] = {0, 0};
            ret = clEnqueueNDRangeKernel(command_queue, kernel, sizeof(global_item_size)/sizeof(global_item_size[0]), global_item_offset, 
                                         global_item_size, local_item_size, 0, NULL, NULL); 
            CHECK_OPENCL_ERROR(ret, "clEnqueueNDRangeKernel() failed");

			clFlush(command_queue);
    	    clFinish(command_queue);
			
			rc = clock_gettime(CLOCK_MONOTONIC, &ts_end1);
			if (rc < 0){
				printf("get start time error!\n");
				return -1;
			}

			//show_data((unsigned int *)map_cl_debug_out, 4);
            //cv::Mat dst(image_infos[i]._img.rows, image_infos[i]._img.cols, CV_8UC1, (void*)image_output_char_data);
			//cv::imshow("processed", dst);

	        //map_cl_debug_out = clEnqueueMapBuffer(command_queue, cl_debug_out,CL_TRUE,CL_MAP_READ, 0, (out_memsize + 2),0, NULL,NULL,&ret);
	        image_infos[i].map_buf_out=clEnqueueMapBuffer(command_queue, image_infos[i]._cl_buf_out ,CL_TRUE,CL_MAP_READ, 0, out_memsize,0, NULL,NULL,&ret);
			for(count=0; count < output_data_count; count++){
				image_output_char_data[count] = (unsigned char)((data_type *)image_infos[i].map_buf_out)[count];
			}

            if (cv::waitKey(500) == 'q') 
            	break;
#if 1
			ret = save_data(image_output_char_data, width_col*height_col);
			if(ret<0){
				printf("save data failed!\n");
			}
#endif
	        clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_out,image_infos[i].map_buf_out,0,NULL,NULL);
			//clEnqueueUnmapMemObject(command_queue,cl_debug_out,map_cl_debug_out,0,NULL,NULL);
        }
    }
	rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);
	if (rc < 0){
		printf("get end time error!\n");
		return -1;
	}
	timespec_sub(&ts_end0, &ts_start0);
	total_time0 += ts_end0.tv_nsec;
	avg_time0 = (float)total_time0/(float)1.0;
	printf("avg time 0: %f sec, %f nsec\n", ts_end0.tv_sec/1.0, avg_time0);

	timespec_sub(&ts_end1, &ts_start1);
	total_time1 += ts_end1.tv_nsec;
	avg_time1 = (float)total_time1/(float)1.0;
	printf("avg time 1: %f sec, %f nsec\n", ts_end1.tv_sec/1.0, avg_time1);
	
	timespec_sub(&ts_end, &ts_start);
	total_time += ts_end.tv_nsec;
	avg_time = (float)total_time/(float)1.0;
	printf("avg time: %f sec, %f nsec\n", ts_end.tv_sec/1.0, avg_time);
	
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    for (int i=0; i<IMAGE_NUM; i++) {
        //free(image_infos[i]._buf_out);
        free(buffer_temp);
        free(output_data);
        free(image_int_data);
        free(image_output_char_data);
	    clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_in,image_infos[i].map_buf_in,0,NULL,NULL);
	    clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_out,image_infos[i].map_buf_out,0,NULL,NULL);

        ret = clReleaseMemObject(image_infos[i]._cl_buf_in);
        ret = clReleaseMemObject(image_infos[i]._cl_buf_out);
    }
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}


