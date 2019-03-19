#pragma once

#include "../examples/CUDA/cuda_header.cuh"
#include "../examples/CUDA/cuda_device_utility.cu"

__global__ void TestDeviceLookupUsage(float* result) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	UnitInfoDevice a = device_unit_lookup[id];
	result[id] = a.range;
}

__global__ void TestDevice3DArrayUsage(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map) {
	extern __shared__ Entity unit_list_s[];
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int x = (id % MAP_X_R), y = (id / MAP_X_R);

	if (id < nr_of_units) unit_list_s[id] = device_unit_list_pointer[id];

	__syncthreads();

	char* devPtr = (char*)device_map.ptr;
	size_t pitch = device_map.pitch;
	size_t slicePitch = pitch * MAP_Y_R;	//not required bcs we have depth 1

	char* slice = devPtr + 0;
	float* row = (float*)(slice + y * pitch);

	row[x] = (float)id;
}