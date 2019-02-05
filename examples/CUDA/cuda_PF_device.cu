#pragma once

#include "../examples/CUDA/cuda_header.cuh"

__global__ void TestDevicePFGeneration(float* device_map) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//move lookup to shared

	//do stuff
}