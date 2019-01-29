#pragma once

#include "../examples/CUDA/cuda_header.cuh"

__global__ void TestDevice(int* data) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	data[id] = OtherDeviceFunction(data[id]);
}

__device__ int OtherDeviceFunction(int input) {
	return input + 50;
}