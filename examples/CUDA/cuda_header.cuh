#pragma once

#ifndef CUDA_HEADER
#define CUDA_HEADER

#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#include <unordered_map>
#include <windows.h>

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
//https://devtalk.nvidia.com/default/topic/482986/how-do-you-copy-an-array-into-constant-memory-/
//https://stackoverflow.com/questions/28821743/sharing-roots-and-weights-for-many-gauss-legendre-quadrature-in-gpus/28822918#28822918

#define BLOCK_AMOUNT 1 
#define THREADS_PER_BLOCK 512 //max 1024, should be multiple of warp size (32)
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

struct UnitInfo {
	sc2::UNIT_TYPEID id = sc2::UNIT_TYPEID::INVALID;
	unsigned int device_id = 0;
	float radius = 0;
	float range = 0;
	bool is_flying = false;
	bool can_attack_air = true;
	bool can_attack_ground = true;
};

typedef struct {
	float range;
	float radius;
	bool is_flying;
	bool can_attack_air;
	bool can_attack_ground;
} UnitInfoDevice;	//must be C-style

//DEVICE FUNCTIONS
__global__ void TestDevicePFGeneration(float* device_map);
__global__ void TestDeviceIMGeneration(float* device_map);
__global__ void TestDeviceLookupUsage(float* result);

class CUDA {
public:
	__host__ CUDA(MapStorage* maps, const sc2::ObservationInterface* observations, sc2::DebugInterface* debug);
	__host__ ~CUDA();

	//Initialization (requires cleanup)
	__host__ void InitializeCUDA();
	__host__ void PrintGenInfo();
	__host__ void CreateUnitLookupOnHost();
	__host__ void TransferStaticMapToHost();
	__host__ void AllocateDeviceMemory();
	__host__ void TransferStaticMapToDevice();
	__host__ void TransferUnitLookupToDevice();
	__host__ void TestLookupTable();
	
	//Runtime functionality
	__host__ void Update(clock_t dt_ticks);
	__host__ bool TransferUnitsToDevice();
	__host__ bool TransferDynamicMapToDevice();
	__host__ bool FillDeviceUnitArray();

	//Other functionality
	__host__ bool DeleteAllIMs();

	//Kernel launches
	__host__ void TestRepellingPFGeneration();
	__host__ void TestAttractingPFGeneration(float range, bool is_flying, bool can_attack_air, bool can_attack_ground);
	__host__ void TestIMGeneration(sc2::Point2D destination, bool air_route);

	//Error checking
	__host__ void Check(cudaError_t blob, std::string location = "unknown", bool print_res = false);	//should not be used in release
private:
	MapStorage* map_storage;	//pointer to Starcraft's map & data interface
	const sc2::ObservationInterface* observation;
	sc2::DebugInterface* debug;

	//device memory pointers
	bool* static_map_device_pointer;	//data in map_storage
	bool* dynamic_map_device_pointer;	//data in map_storage
	UnitInfoDevice* unit_lookup_device_pointer;
	//UnitStructInDevice* unit_array_device_pointer;

	//data
	std::vector<UnitInfo> host_unit_info; 
	std::vector<UnitInfoDevice> device_unit_lookup_on_host;
	std::unordered_map<sc2::UNIT_TYPEID, unsigned int> host_unit_transform;
};
#endif