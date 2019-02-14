#pragma once

#include "../examples/CUDA/cuda_header.cuh"

//DEVICE SYMBOL VARIABLES (const & global)
__device__ __constant__ UnitInfoDevice* device_unit_lookup;

__device__ float GetMapValue(cudaPitchedPtr map, int x, int y) {
	char* ptr = (char*)map.ptr;
	size_t pitch = map.pitch;

	return *((float*)((char*)ptr + y * pitch) + x);
}

__device__ void SetMapValue(cudaPitchedPtr map, int x, int y, float value) {
	char* ptr = (char*)map.ptr;
	size_t pitch = map.pitch;

	float* row = (float*)((char*)ptr + y * pitch);
	row[x] = value;
}

__device__ float FloatDistance(int posX1, int posY1, int posX2, int posY2) {
	float a = powf(posX1 - posX2, 2);
	float b = powf(posY1 - posY2, 2);
	return rsqrtf(a + b);
}

__device__ int IntDistance(int posX1, int posY1, int posX2, int posY2) {
	int a = fabsf(posX1 - posX2);
	int b = fabsf(posY1 - posY2);
	return a + b;
}