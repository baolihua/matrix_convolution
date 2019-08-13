#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>


#define MAX_SOURCE_SIZE (0x10000)

const char* image_files[] = {
    "./gray/1.jpg", "./gray/2.jpg", "./gray/3.jpg",
    "./gray/4.jpg", "./gray/5.jpg", "./gray/6.jpg",
    "./gray/7.jpg", "./gray/8.jpg", "./gray/0.jpg",
    "./gray/9.jpg", NULL
};
struct iinfo {
    cv::Mat _img;
    unsigned char* _buf_out;
    cl_mem _cl_buf_in;
    cl_mem _cl_buf_out;
    void * map_buf_in;
    void * map_buf_out;
};
const int IMAGE_NUM = 10;
struct iinfo image_infos[IMAGE_NUM];

#define CHECK_OPENCL_ERROR(status, str) \
	if (status != CL_SUCCESS) printf("%s\n", str)


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
    FILE *fp;
    char *source_str;
    size_t source_size;
 
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
    cl_kernel kernel = clCreateKernel(program, "sobel", &ret);
    //cl_kernel kernel = clCreateKernel(program, "sobel_vertical", &ret);
    //cl_kernel kernel = clCreateKernel(program, "sobel_horizontal", &ret);
    CHECK_OPENCL_ERROR(ret, "clCreateKernel() failed");
    size_t max_dim;
    size_t max_item_size[3] = {1, 1, 1};
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &max_dim, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, max_item_size, NULL);
    printf("max_dim = %d, [%d, %d, %d]\n", (int)max_dim, (int)max_item_size[0], (int)max_item_size[1], (int)max_item_size[2]);

    for (int i=0; i<IMAGE_NUM; i++) {
    	image_infos[i]._img = cv::imread(image_files[i], cv::IMREAD_GRAYSCALE);
        if (image_infos[i]._img.data == NULL) {
	    printf("failed to read image %s\n", image_files[i]);
	    return -1;
        }

    	unsigned int image_size = image_infos[i]._img.cols * image_infos[i]._img.rows * image_infos[i]._img.channels();
        image_infos[i]._buf_out = (unsigned char*)malloc(image_size);
        if (!image_infos[i]._buf_out) {
            printf("malloc image buffer failed, return\n");
            return -1;
        }
        // Create memory buffers on the device for each vector 
        image_infos[i]._cl_buf_in  = clCreateBuffer(context, CL_MEM_READ_WRITE, image_size, NULL, &ret);
        image_infos[i]._cl_buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, image_size, NULL, &ret);

	image_infos[i].map_buf_in=clEnqueueMapBuffer(command_queue, image_infos[i]._cl_buf_in ,CL_TRUE,CL_MAP_WRITE, 0, image_size,0, NULL,NULL,&ret);
        // Copy the lists A and B to their respective memory buffers
	memcpy(image_infos[i].map_buf_in, image_infos[i]._img.data, image_size);
	clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_in,image_infos[i].map_buf_in,0,NULL,NULL);
        //ret = clEnqueueWriteBuffer(command_queue, image_infos[i]._cl_buf_in, CL_TRUE, 0, image_size, image_infos[i]._img.data, 0, NULL, NULL);
        //CHECK_OPENCL_ERROR(ret, "clEnqueueWriteBuffer() failed");
    }

    int loops = 10000;
    while(loops-- > 0) {
        for (int i=0; i<IMAGE_NUM; i++) {
            cv::imshow("origin", image_infos[i]._img);
            // Set the arguments of the kernel
	    cl_uint width  = image_infos[i]._img.cols;
	    cl_uint height = image_infos[i]._img.rows;
            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_infos[i]._cl_buf_in);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_infos[i]._cl_buf_out);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *)&width);
            ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&height);
            CHECK_OPENCL_ERROR(ret, "clSetKernelArg() failed");
 

            printf("w:%d, h:%d, c:%d\n", image_infos[i]._img.cols, image_infos[i]._img.rows, image_infos[i]._img.channels());
            // Execute the OpenCL kernel on the list
            size_t global_item_size[2] = {(size_t)image_infos[i]._img.cols, (size_t)image_infos[i]._img.rows}; // Process the entire lists
            size_t local_item_size[2] = {8, 8}; // Divide work items into groups of 64
            size_t global_item_offset[2] = {0, 0};
            ret = clEnqueueNDRangeKernel(command_queue, kernel, sizeof(global_item_size)/sizeof(global_item_size[0]), global_item_offset, 
                                         global_item_size, local_item_size, 0, NULL, NULL); 
            CHECK_OPENCL_ERROR(ret, "clEnqueueNDRangeKernel() failed");

            // Read the memory buffer C on the device to the local variable C
		
	    clFlush(command_queue);
    	    clFinish(command_queue);
	    image_infos[i].map_buf_out=clEnqueueMapBuffer(command_queue, image_infos[i]._cl_buf_out ,CL_TRUE,CL_MAP_READ, 0, width*height,0, NULL,NULL,&ret);
    
            cv::Mat dst(image_infos[i]._img.rows, image_infos[i]._img.cols, CV_8UC1, (void*)image_infos[i].map_buf_out);
            cv::imshow("processed", dst);

            if (cv::waitKey(500) == 'q')
            	break;
	    clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_out,image_infos[i].map_buf_out,0,NULL,NULL);
        }
    }
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    for (int i=0; i<IMAGE_NUM; i++) {
        free(image_infos[i]._buf_out);
	clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_in,image_infos[i].map_buf_in,0,NULL,NULL);
	clEnqueueUnmapMemObject(command_queue,image_infos[i]._cl_buf_out,image_infos[i].map_buf_out,0,NULL,NULL);

        ret = clReleaseMemObject(image_infos[i]._cl_buf_in);
        ret = clReleaseMemObject(image_infos[i]._cl_buf_out);
    }
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}
