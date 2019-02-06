#pragma once

#ifndef CUDA_OTHER_FUNCTIONALITY_DEVICE_CU
#define CUDA_OTHER_FUNCTIONALITY_DEVICE_CU

#include "../examples/CUDA/cuda_header.cuh"

__global__ void TestDeviceLookupUsage(float* result) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id == 1) printf("\n1");

	//result[id] = device_unit_lookup_first[id].range;

	if (id == 1) printf("2");

	result[id] = device_unit_lookup_second[id].range;	//this shit crashes kernel

	if (id == 1) printf("3\n");
}

#endif