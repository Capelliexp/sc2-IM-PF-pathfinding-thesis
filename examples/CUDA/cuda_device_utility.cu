#pragma once

#include "../examples/CUDA/cuda_header.cuh"

//DEVICE SYMBOL VARIABLES (const & global)
__device__ __constant__ UnitInfoDevice* device_unit_lookup;

__device__ float GetFloatMapPos(float* ptr, size_t pitch, int x, int y) {
	return *((float*)((char*)ptr + y * pitch) + x);
}

__device__ void SetFloatMapPos(float* ptr, size_t pitch, int x, int y, float value) {
	//float* pElement = (float*)((char*)ptr + y * pitch) + x;
	//*pElement = value;

	float* row = (float*)((char*)ptr + y * pitch);
	row[x] = value;
		
	printf("(<%d, %d>, %f)", x, y, value);
}