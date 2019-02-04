#pragma once

#include "../examples/CUDA/cuda_header.cuh"
//#include "../examples/CUDA/map_storage.hpp"

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

	if (map_storage->update_terrain) TransferDynamicMapToDevice();

}

__host__ bool CUDA::InitializeCUDA(MapStorage* maps) {
	std::cout << "Initializing CUDA object" << std::endl;

	map_storage = maps;
	TransferStaticMapToDevice();

	return true;
}

__host__ void CUDA::TestPFGeneration() {
	float* device_map;
	float* new_map = new float[THREADS_IN_GRID];

	cudaMalloc((void**)&device_map, THREADS_IN_GRID * sizeof(float));	//allocate space for map on device

	TransferUnitsToDevice();

	TestDevicePFGeneration<<<BLOCK_AMOUNT, THREADS_PER_BLOCK>>>(device_map);

	cudaMemcpy(new_map, device_map, THREADS_IN_GRID * sizeof(float), cudaMemcpyDeviceToHost);	//transfer map to host
	//the memcpy should copy to a host 2D array directly, not like this!

	//cudaFree(device_map);	//do not free, space will be used next frame

}

__host__ void CUDA::TestIMGeneration() {
	//TransferDynamicMapToDevice();

	//TestDevice << <BLOCK_AMOUNT, THREADS_PER_BLOCK >> > ();

	//cudaMemcpy();
}

__host__ bool CUDA::TransferUnitsToDevice() {

	return true;
}

__host__ bool CUDA::TransferStaticMapToDevice() {

	return true;
}

__host__ bool CUDA::TransferDynamicMapToDevice() {

	return true;
}