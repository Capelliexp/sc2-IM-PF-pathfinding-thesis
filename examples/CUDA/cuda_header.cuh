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

#include "../examples/CUDA/map_storage.hpp"

//https://devtalk.nvidia.com/default/topic/476201/passing-structures-into-cuda-kernels/

#define BLOCK_AMOUNT 3
#define THREADS_PER_BLOCK 128 //max 1024, should be multiple of warp size (32)
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

//DEVICE FUNCTIONS
__global__ void TestDevicePFGeneration(float* device_map);
__global__ void TestDeviceIMGeneration(float* device_map);

struct UnitStructInDevice {
	float size;	//we might be able to combine size & range?
	float range;
	bool hostile;
};

class CUDA {
public:
	__host__ CUDA(MapStorage* maps);
	__host__ ~CUDA();

	__host__ void Update(clock_t dt_ticks);

	__host__ bool InitializeCUDA(MapStorage* maps);

	__host__ bool TransferUnitsToDevice();
	__host__ bool TransferStaticMapToDevice();
	__host__ bool TransferDynamicMapToDevice();

	__host__ void TestPFGeneration();
	__host__ void TestIMGeneration();
private:
	MapStorage* map_storage;

	bool* static_map_device_pointer;
	bool* dynamic_map_device_pointer;

	UnitStructInDevice* unit_array_device_pointer;
	int device_unit_array_length;
	
};
#endif