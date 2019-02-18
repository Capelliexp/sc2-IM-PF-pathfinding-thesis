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
	int x_build = (id % MAP_X), y_build = (id / MAP_X), x_sub = (id % MAP_X_R), y_sub = (id / MAP_X_R);

	char* ground_map = (char*)device_map_ground.ptr;
	char* air_map = (char*)device_map_air.ptr;

	if (id < nr_of_units) unit_list_s[id] = device_unit_list_pointer[id];

	__syncthreads();

	float ground_charge = 0;
	float air_charge = 0;
	float dist = 0;
	for (int i = 0; i < nr_of_units; ++i) {
		UnitInfoDevice unit = device_unit_lookup[unit_list_s[i].id];
		float range_sub = unit.range;

		if ((dist = (FloatDistance(unit_list_s[i].pos.x, unit_list_s[i].pos.y, x_sub, y_sub) + 0.0001)) < range_sub) {
			ground_charge += ((range_sub / dist) * unit.can_attack_ground * unit_list_s[i].enemy);
			air_charge += ((range_sub / dist) * unit.can_attack_air * unit_list_s[i].enemy);
		}
	}
	
	((float*)(ground_map + y_sub * device_map_ground.pitch))[x_sub] = ground_charge;
	((float*)(air_map + y_sub * device_map_ground.pitch))[x_sub] = air_charge;
}
/*
We could rewrite this to run 2 "passes".
* First to find and calc every unit PF-data affecting every tex_coord. 
* Second to add together all the affecting values and write to the array.

The only difference is that all the threads are writing to global memory
at the same time. Might be beneficial?
*/