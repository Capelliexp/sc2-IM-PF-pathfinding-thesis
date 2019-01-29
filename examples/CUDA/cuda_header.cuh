#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_AMOUNT 3
#define THREADS_PER_BLOCK 128 //max 1024, should be multiple of warp size (32)
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

__host__ int* Test(int* data);
__host__ bool InitializeCUDA(int* data);

__global__ void TestDevice(int* data);

__device__ int OtherDeviceFunction(int input);