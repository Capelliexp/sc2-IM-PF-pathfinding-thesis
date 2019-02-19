#pragma once

#include "../examples/CUDA/cuda_header.cuh"
#include "../examples/CUDA/cuda_device_utility.cu"

/*
1080 deviceQuery:
CUDA Driver Version / Runtime Version          10.1 / 10.0
CUDA Capability Major/Minor version number:    6.1
Total amount of global memory:                 8192 MBytes (8589934592 bytes)
(20) Multiprocessors, (128) CUDA Cores/MP:     2560 CUDA Cores
GPU Max Clock rate:                            1848 MHz (1.85 GHz)
Memory Clock rate:                             5005 Mhz
Memory Bus Width:                              256-bit
L2 Cache Size:                                 2097152 bytes
Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       49152 bytes
Total number of registers available per block: 65536
Warp size:                                     32
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
Maximum memory pitch:                          2147483647 bytes
Texture alignment:                             512 bytes
Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
Run time limit on kernels:                     Yes
Integrated GPU sharing Host Memory:            No
Support host page-locked memory mapping:       Yes
Alignment requirement for Surfaces:            Yes
Device has ECC support:                        Disabled
CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
Device supports Unified Addressing (UVA):      Yes
Device supports Compute Preemption:            Yes
Supports Cooperative Kernel Launch:            No
Supports MultiDevice Co-op Kernel Launch:      No
Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 5

970 deviceQuery:
CUDA Driver Version / Runtime Version          10.1 / 10.0
CUDA Capability Major/Minor version number:    5.2
Total amount of global memory:                 4096 MBytes (4294967296 bytes)
(13) Multiprocessors, (128) CUDA Cores/MP:     1664 CUDA Cores
GPU Max Clock rate:                            1317 MHz (1.32 GHz)
Memory Clock rate:                             3505 Mhz
Memory Bus Width:                              256-bit
L2 Cache Size:                                 1835008 bytes
Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       49152 bytes
Total number of registers available per block: 65536
Warp size:                                     32
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
Maximum memory pitch:                          2147483647 bytes
Texture alignment:                             512 bytes
Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
Run time limit on kernels:                     Yes
Integrated GPU sharing Host Memory:            No
Support host page-locked memory mapping:       Yes
Alignment requirement for Surfaces:            Yes
Device has ECC support:                        Disabled
CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
Device supports Unified Addressing (UVA):      Yes
Device supports Compute Preemption:            No
Supports Cooperative Kernel Launch:            No
Supports MultiDevice Co-op Kernel Launch:      No
Device PCI Domain ID / Bus ID / location ID:   0 / 2 / 0
*/

__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air) {
	extern __shared__ Entity unit_list_s[];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;
	int id_global = x + y * blockDim.x;

	//cull threads outside of tex
	if (x > MAP_X_R || y > MAP_Y_R) return;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	__syncthreads();

	//add upp all fields affecting the owned coord to ground_charge and air_charge
	float ground_charge = 0;
	float air_charge = 0;
	float dist = 0;
	for (int i = 0; i < nr_of_units; ++i) {
		UnitInfoDevice unit = device_unit_lookup[unit_list_s[i].id];
		float range_sub = unit.range;

		if ((dist = (FloatDistance(unit_list_s[i].pos.x, unit_list_s[i].pos.y, x, y) + 0.0001)) < range_sub) {
			ground_charge += ((range_sub / dist) * unit.can_attack_ground * unit_list_s[i].enemy);
			air_charge += ((range_sub / dist) * unit.can_attack_air * unit_list_s[i].enemy);
		}
	}

	//__syncthreads();
	
	//write ground_charge and air_charge to global memory in owned coord
	((float*)(((char*)device_map_ground.ptr) + y * device_map_ground.pitch))[x] = ground_charge;
	((float*)(((char*)device_map_air.ptr) + y * device_map_ground.pitch))[x] = air_charge;
}