#pragma once

#include "../examples/CUDA/cuda_header.cuh"
//#include "../examples/CUDA/cuda_functionality.cu"

//DEVICE SYMBOL VARIABLES
__device__ __constant__ UnitInfoDevice* device_unit_lookup;

__global__ void TestDeviceLookupUsage(float* result) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	result[id] = device_unit_lookup[id].range;
}

__global__ void TestDevicePFGeneration(float* device_map) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//move lookup to shared

	//do stuff
}

__global__ void TestDeviceIMGeneration(float* device_map) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
}
