#pragma once

#include "../examples/CUDA/cuda_header.cuh"

__global__ void TestDeviceLookupUsage(UnitInfoDevice* lookup, float* result) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	result[id] = lookup[id].range;
}