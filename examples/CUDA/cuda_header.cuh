#pragma once

#ifndef CUDA_HEADER
#define CUDA_HEADER

#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "../examples/CUDA/map_storage.hpp"

#define BLOCK_AMOUNT 3
#define THREADS_PER_BLOCK 128 //max 1024, should be multiple of warp size (32)
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

//DEVICE FUNCTIONS
__global__ void TestDevice(int* data);
__device__ int OtherDeviceFunction(int input);

class MapStorage;

class CUDA {
public:
	__host__ CUDA(MapStorage* maps);
	__host__ ~CUDA();

	__host__ void Update(clock_t dt_ticks);

	__host__ bool InitializeCUDA(MapStorage* maps);

	__host__ bool TransferUnitsToDevice(float* data, int length);
	__host__ bool TransferMapToDevice(bool* map);
	__host__ bool TransferMapToHost(float* data, int length);

	__host__ int* Test(int* data);
private:
	MapStorage* map_storage;
	int* data;

};
#endif