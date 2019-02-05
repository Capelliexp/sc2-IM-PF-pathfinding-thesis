#pragma once

#ifndef CUDA_HEADER
#define CUDA_HEADER

#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#include <unordered_map>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"
#include "../examples/CUDA/map_storage.hpp"

//https://devtalk.nvidia.com/default/topic/476201/passing-structures-into-cuda-kernels/
//https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
//http://www.ce.jhu.edu/dalrymple/classes/602/Class13.pdf
//http://disi.unal.edu.co/~gjhernandezp/HeterParallComp/GPU/memory-hardware.pdf
//https://www.reddit.com/r/CUDA/comments/9fkb6n/fastest_way_to_implement_small_lookup_table/

#define BLOCK_AMOUNT 1
#define THREADS_PER_BLOCK 156 //max 1024, should be multiple of warp size (32)
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

struct UnitInfo {
	sc2::UNIT_TYPEID id;
	unsigned int device_id;
	float size;	//we might be able to combine size & range?
	float range = 0;
	//bool hostile;
	bool is_flying = false;
	bool can_attack_air = true;
	bool can_attack_ground = true;
};

struct UnitInfoDevice {
	float range;
	bool is_flying = false;
	bool can_attack_air = true;
	bool can_attack_ground = true;
};

//DEVICE FUNCTIONS
__global__ void TestDevicePFGeneration(float* device_map);
__global__ void TestDeviceIMGeneration(float* device_map);
__global__ void TestDeviceLookupUsage(UnitInfoDevice* lookup, float* result);


class CUDA {
public:
	__host__ CUDA(MapStorage* maps, const sc2::ObservationInterface* observations);
	__host__ ~CUDA();

	__host__ void Update(clock_t dt_ticks);

	__host__ bool InitializeCUDA();

	__host__ void AllocateDeviceMemory();
	__host__ void CreateDeviceLookup();

	__host__ bool TransferUnitsToDevice();
	__host__ bool TransferStaticMapToDevice();
	__host__ bool TransferDynamicMapToDevice();

	__host__ bool DeleteAllIMs();

	__host__ bool FillDeviceUnitArray();

	__host__ void CUDA::TestLookupTable();
	__host__ void TestRepellingPFGeneration();
	__host__ void TestAttractingPFGeneration(float range, bool is_flying, bool can_attack_air, bool can_attack_ground);
	__host__ void TestIMGeneration(sc2::Point2D destination, bool air_route);
private:
	MapStorage* map_storage;	//pointer to Starcraft's map & data interface
	const sc2::ObservationInterface* observation;

	//device memory pointers
	bool* static_map_device_pointer;	//data in map_storage
	bool* dynamic_map_device_pointer;	//data in map_storage
	UnitInfoDevice* unit_lookup_device_pointer;
	//UnitStructInDevice* unit_array_device_pointer;

	//data
	std::vector<UnitInfo> host_unit_info;
	std::vector<UnitInfoDevice> device_unit_lookup;
	std::unordered_map<sc2::UNIT_TYPEID, unsigned int> host_unit_transform;
};
#endif