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
//https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/

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
	bool is_flying;
	bool can_attack_air;
	bool can_attack_ground;
};

class CUDA {
public:
	__host__ CUDA(MapStorage* maps);
	__host__ ~CUDA();

	__host__ void Update(clock_t dt_ticks);

	__host__ bool InitializeCUDA(MapStorage* maps);

	__host__ void AllocateDeviceMemory();

	__host__ bool TransferUnitsToDevice();
	__host__ bool TransferStaticMapToDevice();
	__host__ bool TransferDynamicMapToDevice();

	__host__ bool DeleteAllIMs();

	__host__ bool FillDeviceUnitArray();

	__host__ void TestRepellingPFGeneration();
	__host__ void TestAttractingPFGeneration(float range, bool is_flying, bool can_attack_air, bool can_attack_ground);
	__host__ void TestIMGeneration(sc2::Point2D destination, bool air_route);
private:
	MapStorage* map_storage;	//pointer to Starcraft's map & data interface

	bool* static_map_device_pointer;
	bool* dynamic_map_device_pointer;

	UnitStructInDevice* unit_array_device_pointer;
	int device_unit_array_length;
};
#endif