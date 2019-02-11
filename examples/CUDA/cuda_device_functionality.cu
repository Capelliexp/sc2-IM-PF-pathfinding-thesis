#pragma once

#include "../examples/CUDA/cuda_header.cuh"

/*
1080 deviceQuery:

Total amount of global memory : 8192 MBytes(8589934592 bytes)
(20) Multiprocessors, (128) CUDA Cores / MP : 2560 CUDA Cores
GPU Max Clock rate : 1848 MHz(1.85 GHz)
Memory Clock rate : 5005 Mhz
Memory Bus Width : 256 - bit
L2 Cache Size : 2097152 bytes
Maximum Texture Dimension Size(x, y, z)         1D = (131072), 2D = (131072, 65536), 3D = (16384, 16384, 16384)
Maximum Layered 1D Texture Size, (num)layers  1D = (32768), 2048 layers
Maximum Layered 2D Texture Size, (num)layers  2D = (32768, 32768), 2048 layers
Total amount of constant memory : 65536 bytes
Total amount of shared memory per block : 49152 bytes
Total number of registers available per block : 65536
Warp size : 32
Maximum number of threads per multiprocessor : 2048
Maximum number of threads per block : 1024
Max dimension size of a thread block(x, y, z) : (1024, 1024, 64)
Max dimension size of a grid size(x, y, z) : (2147483647, 65535, 65535)
Maximum memory pitch : 2147483647 bytes
Texture alignment : 512 bytes
*/

//DEVICE SYMBOL VARIABLES
__device__ __constant__ UnitInfoDevice* device_unit_lookup;
//__device__ __shared__ Entity* device_unit_array;	//probably wrong, needed as argument

__global__ void TestDeviceLookupUsage(float* result) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	result[id] = device_unit_lookup[id].range;
}

__global__ void TestDeviceAttractingPFGeneration(float* device_map) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//move lookup to shared

	//do stuff
}

__global__ void TestDeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map){
	extern __shared__ Entity unit_list_s[];
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id < nr_of_units) unit_list_s[id] = device_unit_list_pointer[id];

	__syncthreads();
}

__global__ void TestDeviceIMGeneration(float* device_map) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
}
