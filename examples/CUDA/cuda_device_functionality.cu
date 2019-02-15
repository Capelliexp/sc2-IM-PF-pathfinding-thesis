#pragma once

#include "../examples/CUDA/cuda_header.cuh"
#include "../examples/CUDA/cuda_device_utility.cu"

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

__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air) {
	extern __shared__ Entity unit_list_s[];
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int x = (id % MAP_X_R), y = (id / MAP_X_R);

	if (id < nr_of_units) unit_list_s[id] = device_unit_list_pointer[id];

	__syncthreads();

	char* devPtr = (char*)device_map_ground.ptr;
	size_t pitch = device_map_ground.pitch;

	float* row = (float*)(devPtr + y * pitch);

	row[x] = (float)id;
}