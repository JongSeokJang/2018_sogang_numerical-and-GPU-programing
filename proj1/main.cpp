//
//  main.cpp
//  Simple_SIMT
//
//  Written for CSEG437/CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2018년 Sogang University. All rights reserved.
//

#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "my_OpenCL_util.h"

//#define COALESCED_GLOBAL_MEMORY_ACCESS  // What happens if this line is commented out?

/*
#ifdef COALESCED_GLOBAL_MEMORY_ACCESS
#define OPENCL_C_PROG_FILE_NAME "simple_kernel.cl"
#define KERNEL_NAME "CombineTwoArrays"
#else
#define OPENCL_C_PROG_FILE_NAME "simple_kernel2.cl"
#define KERNEL_NAME "CombineTwoArrays2"
#endif
*/

#define OPENCL_C_PROG_FILE_NAME "reduction.cl"
#define KERNEL_1D_LOCAL			"Reduction_scalar"
#define KERNEL_1D_GLOBAL		"Reduction_global"
#define KERNEL_2D_LOCAL			"Reduction_scalar_2D"
#define KERNEL_2D_GLOBAL		"Reduction_global_2D"

//////////////////////////////////////////////////////////////////////////
void generate_random_float_array(float *array, int n) {
	srand((unsigned int)201803); // Always the same input data
	for (int i = 0; i < n; i++) {
		array[i] = 3.1415926f*((float)rand() / RAND_MAX);
	}
}

float reduction_on_the_CPU_reduction(float *array, int n) {
	int i, j;
	float sum = 0.0f;

	float *array_b = (float *)malloc(sizeof(float)*n);
	if (array_b == NULL) {
		fprintf(stderr, "+++ Error: cannot allocate memory for array_b...\n");
		exit(EXIT_FAILURE);
	}
	memcpy(array_b, array, sizeof(float)*n);

	for (i = n / 2; i > 0; i >>= 1) {
		for (j = 0; j < i; j++) {
			array_b[j] += array_b[j + i];
		}
	}
	sum = array_b[0];
	free(array_b);
	return sum;
}

float reduction_on_the_CPU_KahanSum(float *array, int n) {
	int i;
	float sum = 0.0f, c = 0.0f, t, y;

	for (i = 0; i < n; i++) {
		y = array[i] - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	return sum;
}

//////////////////////////////////////////////////////////////////////////

#define INDEX_GPU 0
#define INDEX_CPU 1
#define DEVICE_GROUP_SIZE 64

typedef struct _OPENCL_C_PROG_SRC {
	size_t length;
	char *string;
} OPENCL_C_PROG_SRC;

int main(void) {

	float reduction_cpu_ret;
	float reduction_gpu_ret;
	float compute_time;
	float *array_A, *array_B, *array_C;

	cl_int				errcode_ret;
	size_t				n_elements, work_group_size_GPU, work_group_size_CPU;	
	size_t				n_elements_2d[2], work_group_size_GPU_2d[2];
	OPENCL_C_PROG_SRC	prog_src;

	cl_platform_id		platform;
	cl_device_id		devices[2];
	cl_context			context;
	cl_command_queue	cmd_queues[2];
	cl_program			program;
	cl_kernel			kernel[4];
	cl_mem				buffer_A, buffer_B, buffer_C_GPU, buffer_C_2dim;
	cl_event			event_for_timing;

	if (0) {
		// Just to reveal my OpenCl platform...
		show_OpenCL_platform();
		system("puase");
		return 0;
	}

	n_elements = 128 * 1024 * 1024;
	n_elements_2d[0] = 1024;
	n_elements_2d[1] = n_elements / n_elements_2d[0];
	work_group_size_GPU = 64; // What would happen if it is 2, 4, 8, 16, 32, 64, 128, 512 or 1024?
	work_group_size_CPU = 64; // What would happen if it is 2, 4, 8, 16, 32, 64, 128, 512 or 1024?
	work_group_size_GPU_2d[0] = 8;
	work_group_size_GPU_2d[1] = work_group_size_GPU / work_group_size_GPU_2d[0];


	array_A = (float *)malloc(sizeof(float)*n_elements);
	array_B = (float *)malloc(sizeof(float)*n_elements);
	array_C = (float *)malloc(sizeof(float)*n_elements / work_group_size_GPU); 

	fprintf(stdout, "^^^ Generating random input arrays with %d elements each...\n", (int)n_elements);
	generate_random_float_array(array_A, (int)n_elements);
	generate_random_float_array(array_B, (int)n_elements);
	memset(array_C, 0x00, sizeof(float)*n_elements / work_group_size_GPU);
	fprintf(stdout, "^^^ Done!\n");


	fprintf(stdout, "n_elements : %d, n_elements_2D : [%d][%d]\n", n_elements, n_elements_2d[0], n_elements_2d[1]);
	fprintf(stdout, "work_group_size : %d, work_group_size_2D : [%d][%d]\n", work_group_size_GPU, work_group_size_GPU_2d[0], work_group_size_GPU_2d[1]);
	fprintf(stdout, "---------------------------------------------------------\n");

	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	fprintf(stdout, "\n^^^ Test 1: general CPU computation(reduction_on_the_CPU_reduction) ^^^\n");
	fprintf(stdout, "   [CPU Execution] \n");
	CHECK_TIME_START;
	//combine_two_arrays_CPU(array_A, array_B, array_C, (int)n_elements);
	reduction_cpu_ret = reduction_on_the_CPU_KahanSum(array_A, (int)n_elements);
	CHECK_TIME_END(compute_time);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	fprintf(stdout, "   [Check Results] \n");
	fprintf(stdout, " [%lf]\n", reduction_cpu_ret);

	///////////////////////////////////////////////////////////////////////////
	fprintf(stdout, "\n^^^ Test 1-1: general CPU computation(reduction_on_the_CPU_KahanSum) ^^^\n");
	fprintf(stdout, "   [CPU Execution] \n");
	CHECK_TIME_START;
	//combine_two_arrays_CPU(array_A, array_B, array_C, (int)n_elements);
	reduction_cpu_ret = reduction_on_the_CPU_KahanSum(array_A, (int)n_elements);
	CHECK_TIME_END(compute_time);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	fprintf(stdout, "   [Check Results] \n");
	fprintf(stdout, " [%lf]\n", reduction_cpu_ret);
	fprintf(stdout, "---------------------------------------------------------\n");


	////////////////////////////////////////////////////////////////////////////////////////

	errcode_ret = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR_CODE(errcode_ret);  
	errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &devices[INDEX_GPU], NULL);
	CHECK_ERROR_CODE(errcode_ret);

	//fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
	//print_device_0(devices[INDEX_GPU]);

	/* Get the first CPU device. */
	errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &devices[INDEX_CPU], NULL);
	CHECK_ERROR_CODE(errcode_ret);

	//fprintf(stdout, "\n^^^ The first CPU device on the platform ^^^\n");
	//print_device_0(devices[INDEX_CPU]);

	context = clCreateContext(NULL, 1, devices, NULL, NULL, &errcode_ret);

	cmd_queues[INDEX_GPU] = clCreateCommandQueue(context, devices[INDEX_GPU], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);

	prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME, &prog_src.string);
	program = clCreateProgramWithSource(context, 1, (const char **)&prog_src.string, &prog_src.length, &errcode_ret);

	errcode_ret = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	if (errcode_ret != CL_SUCCESS) {
		print_build_log(program, devices[INDEX_GPU], "GPU");
		//print_build_log(program, devices[INDEX_CPU], "CPU");
		exit(-1);
	}


	kernel[0] = clCreateKernel(program, KERNEL_1D_LOCAL,	&errcode_ret);
	kernel[1] = clCreateKernel(program, KERNEL_1D_GLOBAL,	&errcode_ret);
	kernel[2] = clCreateKernel(program, KERNEL_2D_LOCAL,	&errcode_ret);
	kernel[3] = clCreateKernel(program, KERNEL_2D_GLOBAL,	&errcode_ret);
	

	// 1-d
	buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n_elements, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	buffer_C_GPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_elements / work_group_size_GPU, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	// 2-d
	buffer_B = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*n_elements, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	buffer_C_2dim = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_elements / work_group_size_GPU, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);


	// 1-d
	// 1-d
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////

	
	fprintf(stdout, "---------------------------------------------------------\n");
	fprintf(stdout, "   [Data Transfer to GPU] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues[INDEX_GPU], buffer_A, CL_FALSE, 0,
		sizeof(float)*n_elements, array_A, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues[INDEX_GPU]); // What if this line is removed?
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	fprintf(stdout, "---------------------------------------------------------\n");

	//local arg set
	errcode_ret = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &buffer_A);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clSetKernelArg(kernel[0], 1, sizeof(float) * work_group_size_GPU, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &buffer_C_GPU);
	CHECK_ERROR_CODE(errcode_ret);

	// global arg set
	errcode_ret = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &buffer_A);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &buffer_C_GPU);
	CHECK_ERROR_CODE(errcode_ret);


	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	fprintf(stdout, "\n^^^ Test 2: Computing on OpenCL GPU_1D Local ^^^\n");

	//printf_KernelWorkGroupInfo(kernel[0], devices[INDEX_GPU]);

	fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[INDEX_GPU], kernel[0], 1, NULL,
		&n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues[INDEX_GPU]);  

	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Data Transfer] \n");

	CHECK_TIME_START;

	errcode_ret = clEnqueueReadBuffer(cmd_queues[INDEX_GPU], buffer_C_GPU, CL_TRUE, 0,
		sizeof(float)*n_elements / work_group_size_GPU, array_C, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);
	// In this case, you do not need to call clFinish() for a synchronization.

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Check Results] \n");
	reduction_gpu_ret = 0;
	reduction_gpu_ret = reduction_on_the_CPU_KahanSum(array_C, (int)n_elements / work_group_size_GPU);
	fprintf(stdout, " [%f]\n", reduction_gpu_ret);
	fprintf(stdout, "---------------------------------------------------------\n");
	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	fprintf(stdout, "\n^^^ Test 3: Computing on OpenCL GPU_1D global ^^^\n");

	//printf_KernelWorkGroupInfo(kernel[1], devices[INDEX_GPU]);

	fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[INDEX_GPU], kernel[1], 1, NULL,
		&n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues[INDEX_GPU]);  // What would happen if this line is removed?
									  // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);


	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Data Transfer] \n");

	CHECK_TIME_START;

	memset(array_C, 0x00, sizeof(float)*n_elements / work_group_size_GPU);

	errcode_ret = clEnqueueReadBuffer(cmd_queues[INDEX_GPU], buffer_C_GPU, CL_TRUE, 0,
		sizeof(float)*n_elements / work_group_size_GPU, array_C, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);
	// In this case, you do not need to call clFinish() for a synchronization.

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Check Results] \n");

	reduction_gpu_ret = 0;
	reduction_gpu_ret = reduction_on_the_CPU_KahanSum(array_C, (int)n_elements / work_group_size_GPU);
	fprintf(stdout, " [%f]\n", reduction_gpu_ret);

	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	// 2-d
	// 2-d

	fprintf(stdout, "---------------------------------------------------------\n");
	fprintf(stdout, "   [Data Transfer to GPU] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues[INDEX_GPU], buffer_B, CL_FALSE, 0,
		sizeof(float)*n_elements, array_B, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues[INDEX_GPU]); // What if this line is removed?
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	fprintf(stdout, "---------------------------------------------------------\n");
	
	// 2d-global
	errcode_ret = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &buffer_B);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clSetKernelArg(kernel[2], 1, sizeof(float) * work_group_size_GPU, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &buffer_C_2dim);
	CHECK_ERROR_CODE(errcode_ret);


	
	errcode_ret = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &buffer_B);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &buffer_C_2dim);
	CHECK_ERROR_CODE(errcode_ret);
	
	fprintf(stdout, "---------------------------------------------------------\n");
	///////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	fprintf(stdout, "\n^^^ Test 4: Computing on OpenCL GPU_2D local ^^^\n");

	//printf_KernelWorkGroupInfo(kernel[2], devices[INDEX_GPU]);

	fprintf(stdout, "   [Kernel Execution] \n");

	/* Execute the kernel on the device. */
	// Enqueues a command to execute a kernel on a device.
	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[INDEX_GPU], kernel[2], 2, NULL,
		n_elements_2d, work_group_size_GPU_2d, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues[INDEX_GPU]);  // What would happen if this line is removed?
									  // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);


	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Data Transfer] \n");

	/* Read back the device buffer to the host array. */
	// Enqueue commands to read from a buffer object to host memory.
	CHECK_TIME_START;
	
	memset(array_C, 0x00, sizeof(float)*n_elements / work_group_size_GPU);

	errcode_ret = clEnqueueReadBuffer(cmd_queues[INDEX_GPU], buffer_C_2dim, CL_TRUE, 0,
		sizeof(float)*n_elements / work_group_size_GPU, array_C, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);
	// In this case, you do not need to call clFinish() for a synchronization.

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Check Results] \n");
	reduction_gpu_ret = 0;
	

	reduction_gpu_ret = reduction_on_the_CPU_KahanSum(array_C, (int)n_elements / work_group_size_GPU);
	fprintf(stdout, " [%f]\n", reduction_gpu_ret);
	fprintf(stdout, "---------------------------------------------------------\n");
	/////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////
	
	fprintf(stdout, "\n^^^ Test 5: Computing on OpenCL GPU_2D global ^^^\n");


	//printf_KernelWorkGroupInfo(kernel[3], devices[INDEX_GPU]);

	fprintf(stdout, "   [Kernel Execution] \n");

	/* Execute the kernel on the device. */
	// Enqueues a command to execute a kernel on a device.
	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[INDEX_GPU], kernel[3], 2, NULL,
		n_elements_2d, work_group_size_GPU_2d, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues[INDEX_GPU]);  // What would happen if this line is removed?
									  // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);


	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Data Transfer] \n");

	/* Read back the device buffer to the host array. */
	// Enqueue commands to read from a buffer object to host memory.
	CHECK_TIME_START;


	memset(array_C, 0x00, sizeof(float)*n_elements / work_group_size_GPU);

	errcode_ret = clEnqueueReadBuffer(cmd_queues[INDEX_GPU], buffer_C_2dim, CL_TRUE, 0,
		sizeof(float)*n_elements / work_group_size_GPU, array_C, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);
	// In this case, you do not need to call clFinish() for a synchronization.

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Check Results] \n");
	reduction_gpu_ret = 0;
	reduction_gpu_ret = reduction_on_the_CPU_KahanSum(array_C, (int)n_elements / work_group_size_GPU);
	fprintf(stdout, " [%f]\n", reduction_gpu_ret);
	fprintf(stdout, "---------------------------------------------------------\n");
	/////////////////////////
	/////////////////////////

	/* Free OpenCL resources. */
	clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_B);
	clReleaseMemObject(buffer_C_GPU);
	clReleaseMemObject(buffer_C_2dim);
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseKernel(kernel[2]);
	clReleaseKernel(kernel[3]);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queues[INDEX_GPU]);
	clReleaseContext(context);

	/* Free host resources. */
	free(array_A);
	free(array_B);
	free(array_C);
	free(prog_src.string);

	system("pause");
	return 0;
}



