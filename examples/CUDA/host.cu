#include "CUDA/device.cu"

#include <stdio.h>
#include <string>
#include <iostream>

__host__ bool InitializeCUDA(int* data) {
	int* new_data = Test(data);

	std::cout << data: << std::endl;
	for (int i = 0; i < THREADS_IN_GRID; ++i) {
		std::cout << new_data[i] << ", ";
	}
	std::cout << std::endl;
}

__host__ void Test(int* data) {
	int* new_data = new int[THREADS_IN_GRID];
	int* device_data = 0;

	cudaMalloc((void**)&device_data, THREADS_IN_GRID * sizeof(int)));
	cudaMemcpy(device_data, data, THREADS_IN_GRID * sizeof(int), cudaMemcpyHostToDevice);

	TestDevice<<<BLOCK_AMOUNT, THREADS_PER_BLOCK>>>(device_data);

	cudaMemcpy(new_data, device_data, THREADS_IN_GRID * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_data);

	return new_data;
}