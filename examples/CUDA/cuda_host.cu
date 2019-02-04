#pragma once

#include "../examples/CUDA/cuda_header.cuh"

#include <stdio.h>
#include <string>
#include <iostream>

__host__ CUDA::CUDA(MapStorage* maps) {
	if (!InitializeCUDA(maps))
		std::cout << "shit b fucked yo" << std::endl;
}

__host__ CUDA::~CUDA() {

}

__host__ void CUDA::Update(clock_t dt_ticks) {
	//float dt = ((float)dt_ticks) / CLOCKS_PER_SEC;	//get dt in seconds

}

__host__ bool CUDA::InitializeCUDA(MapStorage* maps) {
	std::cout << "Initializing CUDA object" << std::endl;

	map_storage = maps;

	data = new int[THREADS_IN_GRID];
	for (int i = 0; i < THREADS_IN_GRID; ++i) data[i] = i;

	//---

	int* new_data = Test(data);

	std::cout << "data:" << std::endl;
	for (int i = 0; i < THREADS_IN_GRID; ++i) {
		std::cout << new_data[i] << ", ";
	}
	std::cout << std::endl;

	return true;
}

__host__ int* CUDA::Test(int* data) {
	int* new_data = new int[THREADS_IN_GRID];
	int* device_data = 0;

	cudaMalloc((void**)&device_data, THREADS_IN_GRID * sizeof(int));
	cudaMemcpy(device_data, data, THREADS_IN_GRID * sizeof(int), cudaMemcpyHostToDevice);

	TestDevice << <BLOCK_AMOUNT, THREADS_PER_BLOCK >> > (device_data);

	cudaMemcpy(new_data, device_data, THREADS_IN_GRID * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_data);

	return new_data;
}

__host__ bool CUDA::TransferUnitsToDevice(float* data, int length) {

	return true;
}

__host__ bool CUDA::TransferMapToDevice(bool* map) {

	return true;
}

__host__ bool CUDA::TransferMapToHost(float* data, int length) {

	return true;
}